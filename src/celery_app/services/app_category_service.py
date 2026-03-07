"""App category lookup and caching service.

This module provides efficient app categorization using:
1. In-memory cache for common apps (fastest)
2. Database cache for previously classified apps (fast)
3. LLM fallback for unknown apps (slower but accurate)
"""

import asyncio
import logging
from datetime import datetime
from typing import Optional, NamedTuple
from zoneinfo import ZoneInfo

from supabase import Client

from src.database import get_supabase_client
from src.llm_utils.services import query_llm

logger = logging.getLogger(__name__)

CHINA_TZ = ZoneInfo("Asia/Shanghai")


class AppCategoryResult(NamedTuple):
    """Result of app category classification."""
    app_name: str
    category: str
    source: str  # 'lookup', 'db_cache', or 'llm'

# =============================================================================
# Predefined App Categories (In-Memory Cache)
# =============================================================================

# Common apps with predefined categories (seed data)
# These are the most popular apps that don't need LLM classification
APP_CATEGORY_CACHE = {
    # Social communication
    "com.whatsapp": "social communication app",
    "com.whatsapp.w4w": "social communication app",
    "com.facebook.katana": "social communication app",
    "com.facebook.orca": "social communication app",  # Messenger
    "com.instagram.android": "social communication app",
    "com.tencent.mm": "social communication app",  # WeChat
    "com.tencent.mobileqq": "social communication app",  # QQ
    "com.tencent.tim": "social communication app",
    "com.ss.android.ugc.aweme": "social communication app",  # TikTok/Douyin
    "com.zhiliaoapp.musically": "social communication app",  # TikTok
    "com.snapchat.android": "social communication app",
    "com.twitter.android": "social communication app",
    "com.linkedin.android": "social communication app",
    "org.telegram.messenger": "social communication app",
    "com.discord": "social communication app",
    "com.signal": "social communication app",
    "com.viber.voip": "social communication app",
    "com.linecorp.line": "social communication app",

    # Video and Music
    "com.youtube.android": "video and music app",
    "com.google.android.youtube": "video and music app",
    "com.netflix.mediaclient": "video and music app",
    "com.spotify.music": "video and music app",
    "com.spotify": "video and music app",
    "com.tencent.qqlive": "video and music app",  # Tencent Video
    "com.youku.phone": "video and music app",  # Youku
    "com.duowan.kiwi": "video and music app",  # Huya
    "com.smile.gifmaker": "video and music app",  # Kuaishou
    "com.netease.cloudmusic": "video and music app",  # NetEase Music
    "com.kugou.android": "video and music app",  # Kugou Music
    "com.tencent.karaoke": "video and music app",  # WeSing
    "com.apple.music": "video and music app",
    "com.amazon.mp3": "video and music app",

    # Games and Gaming
    "com.tencent.tmgp.sgame": "games or gaming platform",  # Honor of Kings
    "com.tencent.tmgp.pubgmhd": "games or gaming platform",  # PUBG Mobile
    "com.tencent.ldj": "games or gaming platform",  # Game for Peace
    "com.mojang.minecraftpe": "games or gaming platform",
    "com.roblox.client": "games or gaming platform",
    "com.supercell.clashofclans": "games or gaming platform",
    "com.supercell.clashroyale": "games or gaming platform",
    "com.ea.game.fifa": "games or gaming platform",
    "com.activision.callofduty": "games or gaming platform",

    # Shopping/E-commerce
    "com.alibaba.android.rimet": "e-commerce/shopping platform",
    "com.taobao.taobao": "e-commerce/shopping platform",
    "com.tmall.wireless": "e-commerce/shopping platform",
    "com.jingdong.app.mall": "e-commerce/shopping platform",  # JD
    "com.pinduoduo": "e-commerce/shopping platform",
    "com.xunmeng.pinduoduo": "e-commerce/shopping platform",
    "com.amazon.mShop.android.shopping": "e-commerce/shopping platform",
    "com.amazon.mobile.shopping": "e-commerce/shopping platform",
    "com.ebay.mobile": "e-commerce/shopping platform",
    "com.alibaba.aliexpress": "e-commerce/shopping platform",
    "me.ele": "e-commerce/shopping platform",  # Ele.me
    "com.sankuai.meituan": "e-commerce/shopping platform",  # Meituan
    "com.sankuai.meituan.takeoutnew": "e-commerce/shopping platform",

    # Office/Working
    "com.microsoft.teams": "office/working app",
    "com.microsoft.teams2": "office/working app",
    "com.microsoft.office.outlook": "office/working app",
    "com.microsoft.office.word": "office/working app",
    "com.microsoft.office.excel": "office/working app",
    "com.microsoft.office.powerpoint": "office/working app",
    "com.microsoft.todos": "office/working app",
    "com.google.android.gm": "office/working app",
    "com.google.android.calendar": "office/working app",
    "com.google.android.apps.docs": "office/working app",
    "com.google.android.apps.docs.editors.docs": "office/working app",
    "com.google.android.apps.docs.editors.sheets": "office/working app",
    "com.google.android.apps.docs.editors.slides": "office/working app",
    "com.tencent.wework": "office/working app",  # WeCom
    "com.tencent.docs": "office/working app",
    "com.feichen.feishu": "office/working app",  # Feishu/Lark
    "com.ss.android.lark": "office/working app",
    "com.atlassian.jira": "office/working app",
    "com.slack": "office/working app",
    "com.zoom.videomeetings": "office/working app",
    "us.zoom.videomeetings": "office/working app",

    # Learning and Education
    "com.duolingo": "learning and education app",
    "com.zhihu.android": "learning and education app",  # Zhihu
    "com.coursera.android": "learning and education app",
    "org.khanacademy.android": "learning and education app",
    "com.udemy.android": "learning and education app",
    "com.brainly": "learning and education app",
    "com.chaoxing": "learning and education app",  # Chaoxing/Xuexitong
    "cn.com.open.mooc": "learning and education app",
    "com.wonder.legal": "learning and education app",

    # Health/Fitness
    "com.fitbit.FitbitMobile": "health management/self-discipline app",
    "com.nike.ntc": "health management/self-discipline app",
    "com.Keep.android": "health management/self-discipline app",
    "com.youdao.calories": "health management/self-discipline app",
    "com.myfitnesspal.android": "health management/self-discipline app",
    "com.strava": "health management/self-discipline app",
    "com.runtastic.android": "health management/self-discipline app",
    "com.dji.health": "health management/self-discipline app",
    "com.apple.health": "health management/self-discipline app",

    # Financial Services
    "com.alipay.android": "financial services app",
    "com.eg.android.AlipayGphone": "financial services app",
    "com.tencent.mm.plugin.wallet": "financial services app",
    "com.icbc": "financial services app",  # ICBC
    "com.ccb": "financial services app",  # CCB
    "com.boc": "financial services app",  # Bank of China
    "com.cmbchina": "financial services app",  # China Merchants Bank
    "com.chase": "financial services app",
    "com.paypal.android.p2pmobile": "financial services app",
    "com.venmo": "financial services app",

    # News/Reading
    "com.netease.news": "news/reading app",
    "com.sina.news": "news/reading app",
    "com.tencent.news": "news/reading app",
    "com.toutiao": "news/reading app",  # Toutiao
    "com.ss.android.article.news": "news/reading app",
    "com.nytimes.android": "news/reading app",
    "com.bbc.news": "news/reading app",
    "com.flipboard": "news/reading app",
    "com.inoreader": "news/reading app",

    # Tools/Utilities
    "com.google.android.googlequicksearchbox": "tool/engineering/functional app",
    "com.google.android.apps.maps": "tool/engineering/functional app",
    "com.google.android.apps.nav": "tool/engineering/functional app",
    "com.autonavi.minimap": "tool/engineering/functional app",  # Amap/Gaode
    "com.baidu.map": "tool/engineering/functional app",
    "com.sogou.map": "tool/engineering/functional app",
    "com.google.android.apps.photos": "tool/engineering/functional app",
    "com.adobe.lrmobile": "tool/engineering/functional app",
    "com.adobe.photoshop": "tool/engineering/functional app",
    "com.microsoft.translator": "tool/engineering/functional app",
    "com.google.android.apps.translate": "tool/engineering/functional app",
    "com.android.chrome": "tool/engineering/functional app",
    "com.chrome.beta": "tool/engineering/functional app",
    "com.UCMobile": "tool/engineering/functional app",
    "com.UCMobile.intl": "tool/engineering/functional app",
    "com.tencent.qqbrowser": "tool/engineering/functional app",
    "com.baidu.searchbox": "tool/engineering/functional app",
    "com.cleanmaster.mguard": "tool/engineering/functional app",
    "com.flutter.forex": "tool/engineering/functional app",
}

# Valid app categories (must match LLM output)
APP_CATEGORIES = [
    "social communication app",
    "common life app",
    "office/working app",
    "learning and education app",
    "e-commerce/shopping platform",
    "news/reading app",
    "video and music app",
    "health management/self-discipline app",
    "financial services app",
    "comprehensive entertainment app",
    "games or gaming platform",
    "tool/engineering/functional app",
    "uncertain",
]


# =============================================================================
# Database Operations
# =============================================================================


async def lookup_app_category_in_db(app_name: str, client: Client) -> Optional[str]:
    """Look up app category in database cache.

    Args:
        app_name: App package name
        client: Supabase client

    Returns:
        Category string if found, None otherwise
    """
    try:
        response = await asyncio.to_thread(
            lambda: client.table("app_categories")
            .select("category")
            .eq("app_name", app_name)
            .limit(1)
            .execute()
        )
        if response.data:
            return response.data[0].get("category")
    except Exception as e:
        logger.debug(f"Error looking up app category in DB: {e}")
    return None


async def cache_app_category_in_db(
    app_name: str,
    category: str,
    source: str,
    client: Client,
) -> bool:
    """Cache app category in database for future lookups.

    Args:
        app_name: App package name
        category: Category string
        source: Source of classification ('lookup' or 'llm')
        client: Supabase client

    Returns:
        True if cached successfully
    """
    try:
        await asyncio.to_thread(
            lambda: client.table("app_categories")
            .upsert({
                "app_name": app_name,
                "category": category,
                "source": source,
                "created_at": datetime.now(CHINA_TZ).isoformat(),
            }, on_conflict="app_name")
            .execute()
        )
        return True
    except Exception as e:
        logger.warning(f"Error caching app category: {e}")
        return False


# =============================================================================
# LLM Classification
# =============================================================================


async def classify_app_via_llm(app_name: str) -> Optional[str]:
    """Classify app using LLM.

    Args:
        app_name: App package name

    Returns:
        Category string from LLM
    """
    system_prompt = """You are an app usage analyst. Categorize the user's current app usage into one of these categories:
- social communication app (Facebook, Instagram, WhatsApp, etc.)
- common life app (daily utilities)
- office/working app (Email, Calendar, Office apps, etc.)
- learning and education app (courses, learning apps)
- e-commerce/shopping platform (Amazon, eBay, food delivery)
- news/reading app (News apps, RSS readers)
- video and music app (YouTube, Netflix, Spotify)
- health management/self-discipline app (Fitness trackers, health apps)
- financial services app (Banking, payment apps)
- comprehensive entertainment app (Games, entertainment)
- games or gaming platform
- tool/engineering/functional app (Maps, GPS, utilities)
- uncertain

Return only the category name exactly as listed above."""

    user_prompt = f"What category best describes this app: {app_name}?"

    try:
        result = await query_llm(system_prompt, user_prompt, temperature=0.1)
        category = result.strip().lower()

        # Validate category is in the expected list
        for valid_cat in APP_CATEGORIES:
            if valid_cat in category:
                return valid_cat

        # If no match, return as uncertain
        logger.warning(f"LLM returned unexpected category '{category}' for app '{app_name}'")
        return "uncertain"
    except Exception as e:
        logger.error(f"Error classifying app via LLM: {e}")
        return None


# =============================================================================
# Main Service Function
# =============================================================================


async def get_app_category(app_name: str, client: Client | None = None) -> Optional[str]:
    """Get app category using table lookup first, then LLM fallback.

    Priority:
    1. In-memory cache (fastest)
    2. Database cache (fast)
    3. LLM classification (accurate but slower)
    4. Return 'uncertain' as fallback

    Args:
        app_name: App package name (e.g., 'com.whatsapp')
        client: Optional Supabase client

    Returns:
        Category string
    """
    result = await get_app_category_with_details(app_name, client)
    return result.category if result else "uncertain"


async def get_app_category_with_details(
    app_name: str,
    client: Client | None = None,
) -> Optional[AppCategoryResult]:
    """Get app category with full details (app_name, category, source).

    Priority:
    1. In-memory cache (fastest)
    2. Database cache (fast)
    3. LLM classification (accurate but slower)
    4. Return 'uncertain' as fallback

    Args:
        app_name: App package name (e.g., 'com.whatsapp')
        client: Optional Supabase client

    Returns:
        AppCategoryResult with app_name, category, and source
    """
    if not app_name:
        return AppCategoryResult(app_name="", category="uncertain", source="none")

    # Normalize app name
    normalized_name = app_name.strip().lower()

    # 1. Check in-memory cache first (fastest)
    if normalized_name in APP_CATEGORY_CACHE:
        logger.debug(f"Using in-memory cached category for '{normalized_name}'")
        return AppCategoryResult(
            app_name=normalized_name,
            category=APP_CATEGORY_CACHE[normalized_name],
            source="lookup"
        )

    # Create client if not provided
    if client is None:
        client = get_supabase_client()

    # 2. Check database cache (for previously classified apps)
    cached = await lookup_app_category_in_db(normalized_name, client)
    if cached:
        logger.debug(f"Using DB cached category for '{normalized_name}': {cached}")
        return AppCategoryResult(
            app_name=normalized_name,
            category=cached,
            source="db_cache"
        )

    # 3. Fall back to LLM classification
    logger.debug(f"Classifying '{normalized_name}' via LLM")
    category = await classify_app_via_llm(normalized_name)

    if category:
        # 4. Cache result in database for future use
        await cache_app_category_in_db(normalized_name, category, "llm", client)
        return AppCategoryResult(
            app_name=normalized_name,
            category=category,
            source="llm"
        )

    # Fallback to uncertain
    return AppCategoryResult(app_name=normalized_name, category="uncertain", source="none")


async def get_app_categories_batch(
    app_names: list[str],
    client: Client | None = None,
) -> dict[str, str]:
    """Get categories for multiple apps efficiently.

    Batch version that minimizes DB queries and LLM calls.

    Args:
        app_names: List of app package names
        client: Optional Supabase client

    Returns:
        Dict mapping app_name -> category
    """
    if client is None:
        client = get_supabase_client()

    results = {}
    apps_needing_llm = []

    for app_name in app_names:
        if not app_name:
            results[app_name] = "uncertain"
            continue

        normalized = app_name.strip().lower()

        # Check in-memory cache
        if normalized in APP_CATEGORY_CACHE:
            results[app_name] = APP_CATEGORY_CACHE[normalized]
        else:
            apps_needing_llm.append(normalized)

    # Batch lookup in database for apps not in memory
    if apps_needing_llm:
        try:
            response = await asyncio.to_thread(
                lambda: client.table("app_categories")
                .select("app_name, category")
                .in_("app_name", apps_needing_llm)
                .execute()
            )

            db_results = {r["app_name"]: r["category"] for r in (response.data or [])}

            for app_name in apps_needing_llm:
                if app_name in db_results:
                    results[app_name] = db_results[app_name]
                    # Remove from needs LLM list
                    apps_needing_llm.remove(app_name)
        except Exception as e:
            logger.warning(f"Error in batch DB lookup: {e}")

    # Classify remaining apps via LLM (could be further optimized with batch LLM calls)
    for app_name in apps_needing_llm:
        if app_name not in results:
            category = await classify_app_via_llm(app_name)
            results[app_name] = category or "uncertain"
            # Cache in database
            await cache_app_category_in_db(app_name, results[app_name], "llm", client)

    return results