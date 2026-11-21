// Source exposure: mixed (pinned image and community function boundaries).
// Boundary source: CRASHLOGGER_PDB(59818@2026.10.24)
// Code origin: original guarded static-callback wrapper template.
// GuardedDestroyCallbacks_131d230, methods and referenced symbols: GUESS(low) STANDIN; original owners and signatures unknown.
// Adapted bodies: none
#pragma once

typedef void (*GuardedCallbackFn)(void*);

extern "C" unsigned long long CallbackGuard3356368;
extern "C" GuardedCallbackFn CallbackSlot3356370;
extern "C" void CleanupTarget12a1d30(void* object);
extern "C" void CleanupTarget1303f90(void* object);
extern "C" void CleanupTarget134a320(void* object);
extern "C" void CleanupTarget1359ff0(void* object);
extern "C" void CleanupTarget135c560(void* object);
extern "C" void CleanupTarget135d660(void* object);
extern "C" void CleanupTarget136a0d0(void* object);
extern "C" void CleanupTarget13ae000(void* object);
extern "C" void CleanupTarget13b3a60(void* object);
extern "C" void CleanupTarget1421ed0(void* object);
extern "C" void CleanupTarget1436400(void* object);
extern "C" void CleanupTarget1440750(void* object);
extern "C" void CleanupTarget1441dd0(void* object);
extern "C" void CleanupTarget145a490(void* object);
extern "C" void CleanupTarget1462140(void* object);

struct GuardedDestroyCallbacks_131d230
{
    static void Destroy131d230(void* object);
    static void Destroy131d770(void* object);
    static void Destroy1358c80(void* object);
    static void Destroy1358cc0(void* object);
    static void Destroy135c060(void* object);
    static void Destroy135c0a0(void* object);
    static void Destroy135d240(void* object);
    static void Destroy135d280(void* object);
    static void Destroy1369bd0(void* object);
    static void Destroy1369c10(void* object);
    static void Destroy138b640(void* object);
    static void Destroy138b980(void* object);
    static void Destroy138ba90(void* object);
    static void Destroy138bed0(void* object);
    static void Destroy13adb60(void* object);
    static void Destroy13adba0(void* object);
    static void Destroy13b2ee0(void* object);
    static void Destroy13b3020(void* object);
    static void Destroy1418750(void* object);
    static void Destroy1418790(void* object);
    static void Destroy1436040(void* object);
    static void Destroy1436080(void* object);
    static void Destroy143f4c0(void* object);
    static void Destroy143f500(void* object);
    static void Destroy14403a0(void* object);
    static void Destroy14403e0(void* object);
    static void Destroy1442e20(void* object);
    static void Destroy1442e60(void* object);
    static void Destroy145a0c0(void* object);
    static void Destroy145a100(void* object);
};
