package com.qlh.inference

import com.qlh.inference.network.AuthCapability
import com.qlh.inference.network.AuthSession

data class AuthUiState(
    val capability: AuthCapability? = null,
    val session: AuthSession? = null,
    val loading: Boolean = false,
    val busy: Boolean = false,
    val message: String? = null,
    val error: String? = null,
) {
    val authenticated: Boolean
        get() = session != null
}
