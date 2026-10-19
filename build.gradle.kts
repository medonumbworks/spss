// The Android Gradle Plugin must share a classloader with the Kotlin Gradle plugin, so it is put on the
// root build classpath here. It is skipped with -Pcryptopulse.skipAndroid=true (JVM-only builds that
// cannot reach Google's Maven repository).
buildscript {
    val skipAndroid = (project.findProperty("cryptopulse.skipAndroid") as String?)?.toBoolean() ?: false
    if (!skipAndroid) {
        repositories {
            google()
            mavenCentral()
            gradlePluginPortal()
        }
        dependencies {
            // keep in sync with `agp` in gradle/libs.versions.toml
            classpath("com.android.tools.build:gradle:8.13.2")
        }
    }
}

plugins {
    alias(libs.plugins.kotlin.jvm) apply false
    alias(libs.plugins.kotlin.android) apply false
    alias(libs.plugins.kotlin.serialization) apply false
    alias(libs.plugins.compose.compiler) apply false
}

allprojects {
    group = "dev.cryptopulse"
    version = "0.1.0"
}
