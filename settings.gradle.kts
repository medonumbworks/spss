// CryptoPulse - Bitpanda crypto overview, forecasts and sentiment.
//
// `-Pcryptopulse.skipAndroid=true` builds only the JVM modules (:core and :mcp-server) and
// skips every Google Maven lookup. Use it on machines without the Android SDK or without
// access to dl.google.com (for example locked-down CI runners or cloud sandboxes).
val skipAndroid: Boolean = providers.gradleProperty("cryptopulse.skipAndroid").orNull?.toBoolean() ?: false

pluginManagement {
    repositories {
        val skip = providers.gradleProperty("cryptopulse.skipAndroid").orNull?.toBoolean() ?: false
        if (!skip) {
            google {
                content {
                    includeGroupByRegex("com\\.android.*")
                    includeGroupByRegex("com\\.google.*")
                    includeGroupByRegex("androidx.*")
                }
            }
        }
        gradlePluginPortal()
        mavenCentral()
    }
}

dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        val skip = providers.gradleProperty("cryptopulse.skipAndroid").orNull?.toBoolean() ?: false
        if (!skip) {
            google {
                content {
                    includeGroupByRegex("com\\.android.*")
                    includeGroupByRegex("com\\.google.*")
                    includeGroupByRegex("androidx.*")
                }
            }
        }
        mavenCentral()
    }
}

rootProject.name = "cryptopulse"
include(":core", ":mcp-server")
if (!skipAndroid) {
    include(":app")
}
