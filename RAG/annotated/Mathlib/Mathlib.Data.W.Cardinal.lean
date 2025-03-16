theorem cardinalMk_eq_sum_lift : #(WType β) = sum fun a ↦ #(WType β) ^ lift.{u} #(β a) :=
  (mk_congr <| equivSigma β).trans <| by
    /-
      α : Type u
      β : α → Type v
      ⊢ Eq (Cardinal.mk (Sigma fun a => β a → WType β)) (Cardinal.sum fun a => HPow. …
    -/
    simp_rw [mk_sigma, mk_arrow]; rw [lift_id'.{v, u}, lift_umax.{v, u}]
                                  /-
                                    🎉 no goals
                                  -/


@[deprecated (since := "2024-11-10")] alias cardinal_mk_eq_sum' := cardinalMk_eq_sum_lift


/-- `#(WType β)` is the least cardinal `κ` such that `sum (fun a : α ↦ κ ^ #(β a)) ≤ κ` -/
theorem cardinalMk_le_of_le' {κ : Cardinal.{max u v}}
    (hκ : (sum fun a : α => κ ^ lift.{u} #(β a)) ≤ κ) :
    #(WType β) ≤ κ := by
  /-
    α : Type u
    β : α → Type v
    κ : Cardinal.{max u v}
    hκ : LE.le (Cardinal.sum fun a => HPow.hPow κ (Cardinal.lift.{u, v} (Cardinal. …
    ⊢ LE.le (Cardinal.mk (WType β)) κ
  -/
  induction' κ using Cardinal.inductionOn with γ
  /-
    case h
    α : Type u
    β : α → Type v
    γ : Type (max u v)
    hκ : LE.le (Cardinal.sum fun a => HPow.hPow (Cardinal.mk γ) (Cardinal.lift.{u, …
    ⊢ LE.le (Cardinal.mk (WType β)) (Cardinal.mk γ)
  -/
  simp_rw [← lift_umax.{v, u}] at hκ
  /-
    case h
    α : Type u
    β : α → Type v
    γ : Type (max u v)
    hκ : LE.le (Cardinal.sum fun a => HPow.hPow (Cardinal.mk γ) (Cardinal.lift.{ma …
    ⊢ LE.le (Cardinal.mk (WType β)) (Cardinal.mk γ)
  -/
  nth_rewrite 1 [← lift_id'.{v, u} #γ] at hκ
  /-
    case h
    α : Type u
    β : α → Type v
    γ : Type (max u v)
    hκ : LE.le (Cardinal.sum fun a => HPow.hPow (Cardinal.lift.{v, max v u} (Cardi …
    ⊢ LE.le (Cardinal.mk (WType β)) (Cardinal.mk γ)
  -/
  simp_rw [← mk_arrow, ← mk_sigma, le_def] at hκ
  /-
    case h
    α : Type u
    β : α → Type v
    γ : Type (max u v)
    hκ : Nonempty (Function.Embedding (Sigma fun i => β i → γ) γ)
    ⊢ LE.le (Cardinal.mk (WType β)) (Cardinal.mk γ)
  -/
  cases' hκ with hκ
  /-
    case h.intro
    α : Type u
    β : α → Type v
    γ : Type (max u v)
    hκ : Function.Embedding (Sigma fun i => β i → γ) γ
    ⊢ LE.le (Cardinal.mk (WType β)) (Cardinal.mk γ)
  -/
  exact Cardinal.mk_le_of_injective (elim_injective _ hκ.1 hκ.2)
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-11-10")] alias cardinal_mk_le_of_le' := cardinalMk_le_of_le'


/-- If, for any `a : α`, `β a` is finite, then the cardinality of `WType β`
  is at most the maximum of the cardinality of `α` and `ℵ₀`  -/
theorem cardinalMk_le_max_aleph0_of_finite' [∀ a, Finite (β a)] :
    #(WType β) ≤ max (lift.{v} #α) ℵ₀ :=
  (isEmpty_or_nonempty α).elim
    (by
      /-
        α : Type u
        β : α → Type v
        inst✝ : ∀ (a : α), Finite (β a)
        ⊢ IsEmpty α → LE.le (Cardinal.mk (WType β)) (Max.max (Cardinal.lift.{v, u} (Ca …
      -/
      intro h
      /-
        α : Type u
        β : α → Type v
        inst✝ : ∀ (a : α), Finite (β a)
        h : IsEmpty α
        ⊢ LE.le (Cardinal.mk (WType β)) (Max.max (Cardinal.lift.{v, u} (Cardinal.mk α) …
      -/
      rw [Cardinal.mk_eq_zero (WType β)]
      /-
        α : Type u
        β : α → Type v
        inst✝ : ∀ (a : α), Finite (β a)
        h : IsEmpty α
        ⊢ LE.le 0 (Max.max (Cardinal.lift.{v, u} (Cardinal.mk α)) Cardinal.aleph0)
      -/
      exact zero_le _)
      /-
        🎉 no goals
      -/
    fun hn =>
    let m := max (lift.{v} #α) ℵ₀
    cardinalMk_le_of_le' <|
      calc
        (Cardinal.sum fun a => m ^ lift.{u} #(β a)) ≤ lift.{v} #α * ⨆ a, m ^ lift.{u} #(β a) :=
          Cardinal.sum_le_iSup_lift _
        _ ≤ m * ⨆ a, m ^ lift.{u} #(β a) := mul_le_mul' (le_max_left _ _) le_rfl
        _ = m :=
          mul_eq_left (le_max_right _ _)
              (ciSup_le' fun _ => pow_le (le_max_right _ _) (lt_aleph0_of_finite _)) <|
            pos_iff_ne_zero.1 <|
              Order.succ_le_iff.1
                (by
                  /-
                    α : Type u
                    β : α → Type v
                    inst✝ : ∀ (a : α), Finite (β a)
                    hn : Nonempty α
                    m : Cardinal.{max u v} := Max.max (Cardinal.lift.{v, u} (Cardinal.mk α)) Cardi …
                    ⊢ LE.le (Order.succ 0) (iSup fun a => HPow.hPow m (Cardinal.lift.{u, v} (Cardi …
                  -/
                  rw [succ_zero]
                  /-
                    α : Type u
                    β : α → Type v
                    inst✝ : ∀ (a : α), Finite (β a)
                    hn : Nonempty α
                    m : Cardinal.{max u v} := Max.max (Cardinal.lift.{v, u} (Cardinal.mk α)) Cardi …
                    ⊢ LE.le 1 (iSup fun a => HPow.hPow m (Cardinal.lift.{u, v} (Cardinal.mk (β a))))
                  -/
                  obtain ⟨a⟩ : Nonempty α := hn
                  /-
                    case intro
                    α : Type u
                    β : α → Type v
                    inst✝ : ∀ (a : α), Finite (β a)
                    m : Cardinal.{max u v} := Max.max (Cardinal.lift.{v, u} (Cardinal.mk α)) Cardi …
                    a : α
                    ⊢ LE.le 1 (iSup fun a => HPow.hPow m (Cardinal.lift.{u, v} (Cardinal.mk (β a))))
                  -/
                  refine le_trans ?_ (le_ciSup (bddAbove_range _) a)
                  /-
                    case intro
                    α : Type u
                    β : α → Type v
                    inst✝ : ∀ (a : α), Finite (β a)
                    m : Cardinal.{max u v} := Max.max (Cardinal.lift.{v, u} (Cardinal.mk α)) Cardi …
                    a : α
                    ⊢ LE.le 1 (HPow.hPow m (Cardinal.lift.{u, v} (Cardinal.mk (β a))))
                  -/
                  rw [← power_zero]
                  exact
                    power_le_power_left
                      (pos_iff_ne_zero.1 (aleph0_pos.trans_le (le_max_right _ _))) (zero_le _))


@[deprecated (since := "2024-11-10")]
alias cardinal_mk_le_max_aleph0_of_finite' := cardinalMk_le_max_aleph0_of_finite'


theorem cardinalMk_eq_sum : #(WType β) = sum (fun a : α => #(WType β) ^ #(β a)) :=
                                     /-
                                       α : Type u
                                       β : α → Type u
                                       ⊢ Eq (Cardinal.sum fun a => HPow.hPow (Cardinal.mk (WType β)) (Cardinal.lift.{ …
                                     -/
  cardinalMk_eq_sum_lift.trans <| by simp_rw [lift_id]
                                     /-
                                       🎉 no goals
                                     -/


@[deprecated (since := "2024-11-10")] alias cardinal_mk_eq_sum := cardinalMk_eq_sum


/-- `#(WType β)` is the least cardinal `κ` such that `sum (fun a : α ↦ κ ^ #(β a)) ≤ κ` -/
theorem cardinalMk_le_of_le {κ : Cardinal.{u}} (hκ : (sum fun a : α => κ ^ #(β a)) ≤ κ) :
                                                 /-
                                                   α : Type u
                                                   β : α → Type u
                                                   κ : Cardinal.{u}
                                                   hκ : LE.le (Cardinal.sum fun a => HPow.hPow κ (Cardinal.mk (β a))) κ
                                                   ⊢ LE.le (Cardinal.sum fun a => HPow.hPow κ (Cardinal.lift.{u, u} (Cardinal.mk  …
                                                 -/
    #(WType β) ≤ κ := cardinalMk_le_of_le' <| by simp_rw [lift_id]; exact hκ
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


@[deprecated (since := "2024-11-10")] alias cardinal_mk_le_of_le := cardinalMk_le_of_le


/-- If, for any `a : α`, `β a` is finite, then the cardinality of `WType β`
  is at most the maximum of the cardinality of `α` and `ℵ₀`  -/
theorem cardinalMk_le_max_aleph0_of_finite [∀ a, Finite (β a)] : #(WType β) ≤ max #α ℵ₀ :=
                                                     /-
                                                       α : Type u
                                                       β : α → Type u
                                                       inst✝ : ∀ (a : α), Finite (β a)
                                                       ⊢ Eq (Max.max (Cardinal.lift.{u, u} (Cardinal.mk α)) Cardinal.aleph0) (Max.max …
                                                     -/
  cardinalMk_le_max_aleph0_of_finite'.trans_eq <| by rw [lift_id]
                                                     /-
                                                       🎉 no goals
                                                     -/


@[deprecated (since := "2024-11-10")]
alias cardinal_mk_le_max_aleph0_of_finite := cardinalMk_le_max_aleph0_of_finite


