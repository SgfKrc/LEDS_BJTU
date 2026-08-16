package com.qlh.inference

import com.qlh.inference.network.ConnectionHealthReport

/** State shown by the settings maintenance controls. No credential is kept here. */
data class DiagnosticsUiState(
    val health: ConnectionHealthReport? = null,
    val healthLoading: Boolean = false,
    val healthError: String? = null,
    val uploadInProgress: Boolean = false,
    val uploadMessage: String? = null,
    val uploadError: String? = null,
)
