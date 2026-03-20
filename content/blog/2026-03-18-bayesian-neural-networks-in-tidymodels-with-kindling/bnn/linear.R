box::use(
    torch[
        nn_module, nn_parameter
    ],
    torch[
        torch_empty, torch_randn_like, torch_exp,
        with_no_grad, nnf_linear
    ]
)

#' @export
BayesLinear = nn_module(
    "BayesLinear",
    initialize = function(in_features, out_features, prior_mu, prior_sigma, bias = TRUE) {
        self$in_features = in_features
        self$out_features = out_features

        # ---- Main parameters ----
        self$prior_mu = prior_mu
        self$prior_sigma = prior_sigma
        self$prior_log_sigma = log(prior_sigma)    # base R log(), prior_sigma is a plain numeric

        # ---- Pre-setted parameters ----
        self$weight_mu = nn_parameter(
            torch_empty(out_features, in_features)
        )
        self$weight_log_sigma = nn_parameter(
            torch_empty(out_features, in_features)
        )
        self$register_buffer("weight_eps", NULL)

        self$bias = bias

        # ---- Handling bias parameter ----
        if (self$bias) {
            self$bias_mu = nn_parameter(torch_empty(out_features))
            self$bias_log_sigma = nn_parameter(torch_empty(out_features))
            self$register_buffer("bias_eps", NULL)
        } else {
            self$register_parameter("bias_mu", NULL)
            self$register_parameter("bias_log_sigma", NULL)
            self$register_buffer("bias_eps", NULL)
        }

        self$reset_parameters()
    },
    reset_parameters = function() {
        stdv = 1 / sqrt(self$in_features)
        torch::with_no_grad({
            self$weight_mu$uniform_(-stdv, stdv)
            self$weight_log_sigma$fill_(self$prior_log_sigma)
            if (self$bias) {
                self$bias_mu$uniform_(-stdv, stdv)
                self$bias_log_sigma$fill_(self$prior_log_sigma)
            }
        })
    },
    freeze = function() {
        self$weight_eps = torch_randn_like(self$weight_log_sigma)
        if (self$bias)
            self$bias_eps = torch_randn_like(self$bias_log_sigma)
    },
    unfreeze = function() {
        self$weight_eps = NULL
        if (self$bias)
            self$bias_eps = NULL
    },
    forward = function(x) {
        # ---- Weight calculation ----
        weight = if (is.null(self$weight_eps)) {
            self$weight_mu + torch_exp(self$weight_log_sigma) * torch_randn_like(self$weight_log_sigma)
        } else {
            self$weight_mu + torch_exp(self$weight_log_sigma) * self$weight_eps
        }

        # ---- Bias calculation (if TRUE) ----
        bias = if (self$bias) {
            if (is.null(self$bias_eps)) {
                self$bias_mu + torch_exp(self$bias_log_sigma) * torch_randn_like(self$bias_log_sigma)
            } else {
                self$bias_mu + torch_exp(self$bias_log_sigma) * self$bias_eps
            }
        } else {
            NULL
        }

        nnf_linear(x, weight, bias)
    }
)
