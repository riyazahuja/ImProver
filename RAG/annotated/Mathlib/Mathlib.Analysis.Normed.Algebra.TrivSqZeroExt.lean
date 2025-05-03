local notation "tsze" => TrivSqZeroExt


@[simp] theorem fst_expSeries (x : tsze R M) (n : ℕ) :
    fst (expSeries 𝕜 (tsze R M) n fun _ => x) = expSeries 𝕜 R n fun _ => x.fst := by
  /-
    𝕜 : Type u_1
    R : Type u_3
    M : Type u_4
    inst✝¹⁵ : Field 𝕜
    inst✝¹⁴ : Ring R
    inst✝¹³ : AddCommGroup M
    inst✝¹² : Algebra 𝕜 R
    inst✝¹¹ : Module 𝕜 M
    inst✝¹⁰ : Module R M
    inst✝⁹ : Module (MulOpposite R) M
    inst✝⁸ : SMulCommClass R (MulOpposite R) M
    inst✝⁷ : IsScalarTower 𝕜 R M
    inst✝⁶ : IsScalarTower 𝕜 (MulOpposite R) M
    inst✝⁵ : TopologicalSpace R
    inst✝⁴ : TopologicalSpace M
    inst✝³ : TopologicalRing R
    inst✝² : TopologicalAddGroup M
    inst✝¹ : ContinuousSMul R M
    inst✝ : ContinuousSMul (MulOpposite R) M
    x : TrivSqZeroExt R M
    n : Nat
    ⊢ Eq ((NormedSpace.expSeries 𝕜 (TrivSqZeroExt R M) n) fun x_1 => x).fst ((Norm …
  -/
  simp [expSeries_apply_eq]
  /-
    🎉 no goals
  -/


theorem snd_expSeries_of_smul_comm
    (x : tsze R M) (hx : MulOpposite.op x.fst • x.snd = x.fst • x.snd) (n : ℕ) :
    snd (expSeries 𝕜 (tsze R M) (n + 1) fun _ => x) = (expSeries 𝕜 R n fun _ => x.fst) • x.snd := by
  simp_rw [expSeries_apply_eq, snd_smul, snd_pow_of_smul_comm _ _ hx,
    ← Nat.cast_smul_eq_nsmul 𝕜 (n + 1), smul_smul, smul_assoc, Nat.factorial_succ, Nat.pred_succ,
    Nat.cast_mul, mul_inv_rev,
    inv_mul_cancel_right₀ ((Nat.cast_ne_zero (R := 𝕜)).mpr <| Nat.succ_ne_zero n)]


/-- If `NormedSpace.exp R x.fst` converges to `e`
then `(NormedSpace.exp R x).snd` converges to `e • x.snd`. -/
theorem hasSum_snd_expSeries_of_smul_comm (x : tsze R M)
    (hx : MulOpposite.op x.fst • x.snd = x.fst • x.snd) {e : R}
    (h : HasSum (fun n => expSeries 𝕜 R n fun _ => x.fst) e) :
    HasSum (fun n => snd (expSeries 𝕜 (tsze R M) n fun _ => x)) (e • x.snd) := by
  /-
    𝕜 : Type u_1
    R : Type u_3
    M : Type u_4
    inst✝¹⁶ : Field 𝕜
    inst✝¹⁵ : CharZero 𝕜
    inst✝¹⁴ : Ring R
    inst✝¹³ : AddCommGroup M
    inst✝¹² : Algebra 𝕜 R
    inst✝¹¹ : Module 𝕜 M
    inst✝¹⁰ : Module R M
    inst✝⁹ : Module (MulOpposite R) M
    inst✝⁸ : SMulCommClass R (MulOpposite R) M
    inst✝⁷ : IsScalarTower 𝕜 R M
    inst✝⁶ : IsScalarTower 𝕜 (MulOpposite R) M
    inst✝⁵ : TopologicalSpace R
    inst✝⁴ : TopologicalSpace M
    inst✝³ : TopologicalRing R
    inst✝² : TopologicalAddGroup M
    inst✝¹ : ContinuousSMul R M
    inst✝ : ContinuousSMul (MulOpposite R) M
    x : TrivSqZeroExt R M
    hx : Eq (HSMul.hSMul (MulOpposite.op x.fst) x.snd) (HSMul.hSMul x.fst x.snd)
    e : R
    h : HasSum (fun n => (NormedSpace.expSeries 𝕜 R n) fun x_1 => x.fst) e
    ⊢ HasSum (fun n => ((NormedSpace.expSeries 𝕜 (TrivSqZeroExt R M) n) fun x_1 => …
  -/
  rw [← hasSum_nat_add_iff' 1]
  /-
    𝕜 : Type u_1
    R : Type u_3
    M : Type u_4
    inst✝¹⁶ : Field 𝕜
    inst✝¹⁵ : CharZero 𝕜
    inst✝¹⁴ : Ring R
    inst✝¹³ : AddCommGroup M
    inst✝¹² : Algebra 𝕜 R
    inst✝¹¹ : Module 𝕜 M
    inst✝¹⁰ : Module R M
    inst✝⁹ : Module (MulOpposite R) M
    inst✝⁸ : SMulCommClass R (MulOpposite R) M
    inst✝⁷ : IsScalarTower 𝕜 R M
    inst✝⁶ : IsScalarTower 𝕜 (MulOpposite R) M
    inst✝⁵ : TopologicalSpace R
    inst✝⁴ : TopologicalSpace M
    inst✝³ : TopologicalRing R
    inst✝² : TopologicalAddGroup M
    inst✝¹ : ContinuousSMul R M
    inst✝ : ContinuousSMul (MulOpposite R) M
    x : TrivSqZeroExt R M
    hx : Eq (HSMul.hSMul (MulOpposite.op x.fst) x.snd) (HSMul.hSMul x.fst x.snd)
    e : R
    h : HasSum (fun n => (NormedSpace.expSeries 𝕜 R n) fun x_1 => x.fst) e
    ⊢ HasSum (fun n => ((NormedSpace.expSeries 𝕜 (TrivSqZeroExt R M) (HAdd.hAdd n  …
  -/
  simp_rw [snd_expSeries_of_smul_comm _ _ hx]
  /-
    𝕜 : Type u_1
    R : Type u_3
    M : Type u_4
    inst✝¹⁶ : Field 𝕜
    inst✝¹⁵ : CharZero 𝕜
    inst✝¹⁴ : Ring R
    inst✝¹³ : AddCommGroup M
    inst✝¹² : Algebra 𝕜 R
    inst✝¹¹ : Module 𝕜 M
    inst✝¹⁰ : Module R M
    inst✝⁹ : Module (MulOpposite R) M
    inst✝⁸ : SMulCommClass R (MulOpposite R) M
    inst✝⁷ : IsScalarTower 𝕜 R M
    inst✝⁶ : IsScalarTower 𝕜 (MulOpposite R) M
    inst✝⁵ : TopologicalSpace R
    inst✝⁴ : TopologicalSpace M
    inst✝³ : TopologicalRing R
    inst✝² : TopologicalAddGroup M
    inst✝¹ : ContinuousSMul R M
    inst✝ : ContinuousSMul (MulOpposite R) M
    x : TrivSqZeroExt R M
    hx : Eq (HSMul.hSMul (MulOpposite.op x.fst) x.snd) (HSMul.hSMul x.fst x.snd)
    e : R
    h : HasSum (fun n => (NormedSpace.expSeries 𝕜 R n) fun x_1 => x.fst) e
    ⊢ HasSum (fun n => HSMul.hSMul ((NormedSpace.expSeries 𝕜 R n) fun x_1 => x.fst …
  -/
  simp_rw [expSeries_apply_eq] at *
  rw [Finset.range_one, Finset.sum_singleton, Nat.factorial_zero, Nat.cast_one, pow_zero,
    inv_one, one_smul, snd_one, sub_zero]
  /-
    𝕜 : Type u_1
    R : Type u_3
    M : Type u_4
    inst✝¹⁶ : Field 𝕜
    inst✝¹⁵ : CharZero 𝕜
    inst✝¹⁴ : Ring R
    inst✝¹³ : AddCommGroup M
    inst✝¹² : Algebra 𝕜 R
    inst✝¹¹ : Module 𝕜 M
    inst✝¹⁰ : Module R M
    inst✝⁹ : Module (MulOpposite R) M
    inst✝⁸ : SMulCommClass R (MulOpposite R) M
    inst✝⁷ : IsScalarTower 𝕜 R M
    inst✝⁶ : IsScalarTower 𝕜 (MulOpposite R) M
    inst✝⁵ : TopologicalSpace R
    inst✝⁴ : TopologicalSpace M
    inst✝³ : TopologicalRing R
    inst✝² : TopologicalAddGroup M
    inst✝¹ : ContinuousSMul R M
    inst✝ : ContinuousSMul (MulOpposite R) M
    x : TrivSqZeroExt R M
    hx : Eq (HSMul.hSMul (MulOpposite.op x.fst) x.snd) (HSMul.hSMul x.fst x.snd)
    e : R
    h : HasSum (fun n => HSMul.hSMul (Inv.inv ↑n.factorial) (HPow.hPow x.fst n)) e
    ⊢ HasSum (fun n => HSMul.hSMul (HSMul.hSMul (Inv.inv ↑n.factorial) (HPow.hPow  …
  -/
  exact h.smul_const _
  /-
    🎉 no goals
  -/


/-- If `NormedSpace.exp R x.fst` converges to `e`
then `NormedSpace.exp R x` converges to `inl e + inr (e • x.snd)`. -/
theorem hasSum_expSeries_of_smul_comm
    (x : tsze R M) (hx : MulOpposite.op x.fst • x.snd = x.fst • x.snd)
    {e : R} (h : HasSum (fun n => expSeries 𝕜 R n fun _ => x.fst) e) :
    HasSum (fun n => expSeries 𝕜 (tsze R M) n fun _ => x) (inl e + inr (e • x.snd)) := by
  have : HasSum (fun n => fst (expSeries 𝕜 (tsze R M) n fun _ => x)) e := by
    simpa [fst_expSeries] using h
  simpa only [inl_fst_add_inr_snd_eq] using
    (hasSum_inl _ <| this).add (hasSum_inr _ <| hasSum_snd_expSeries_of_smul_comm 𝕜 x hx h)


theorem exp_def_of_smul_comm (x : tsze R M) (hx : MulOpposite.op x.fst • x.snd = x.fst • x.snd) :
    exp 𝕜 x = inl (exp 𝕜 x.fst) + inr (exp 𝕜 x.fst • x.snd) := by
  /-
    𝕜 : Type u_1
    R : Type u_3
    M : Type u_4
    inst✝¹⁸ : Field 𝕜
    inst✝¹⁷ : CharZero 𝕜
    inst✝¹⁶ : Ring R
    inst✝¹⁵ : AddCommGroup M
    inst✝¹⁴ : Algebra 𝕜 R
    inst✝¹³ : Module 𝕜 M
    inst✝¹² : Module R M
    inst✝¹¹ : Module (MulOpposite R) M
    inst✝¹⁰ : SMulCommClass R (MulOpposite R) M
    inst✝⁹ : IsScalarTower 𝕜 R M
    inst✝⁸ : IsScalarTower 𝕜 (MulOpposite R) M
    inst✝⁷ : TopologicalSpace R
    inst✝⁶ : TopologicalSpace M
    inst✝⁵ : TopologicalRing R
    inst✝⁴ : TopologicalAddGroup M
    inst✝³ : ContinuousSMul R M
    inst✝² : ContinuousSMul (MulOpposite R) M
    inst✝¹ : T2Space R
    inst✝ : T2Space M
    x : TrivSqZeroExt R M
    hx : Eq (HSMul.hSMul (MulOpposite.op x.fst) x.snd) (HSMul.hSMul x.fst x.snd)
    ⊢ Eq (NormedSpace.exp 𝕜 x) (HAdd.hAdd (TrivSqZeroExt.inl (NormedSpace.exp 𝕜 x. …
  -/
  simp_rw [exp, FormalMultilinearSeries.sum]
  /-
    𝕜 : Type u_1
    R : Type u_3
    M : Type u_4
    inst✝¹⁸ : Field 𝕜
    inst✝¹⁷ : CharZero 𝕜
    inst✝¹⁶ : Ring R
    inst✝¹⁵ : AddCommGroup M
    inst✝¹⁴ : Algebra 𝕜 R
    inst✝¹³ : Module 𝕜 M
    inst✝¹² : Module R M
    inst✝¹¹ : Module (MulOpposite R) M
    inst✝¹⁰ : SMulCommClass R (MulOpposite R) M
    inst✝⁹ : IsScalarTower 𝕜 R M
    inst✝⁸ : IsScalarTower 𝕜 (MulOpposite R) M
    inst✝⁷ : TopologicalSpace R
    inst✝⁶ : TopologicalSpace M
    inst✝⁵ : TopologicalRing R
    inst✝⁴ : TopologicalAddGroup M
    inst✝³ : ContinuousSMul R M
    inst✝² : ContinuousSMul (MulOpposite R) M
    inst✝¹ : T2Space R
    inst✝ : T2Space M
    x : TrivSqZeroExt R M
    hx : Eq (HSMul.hSMul (MulOpposite.op x.fst) x.snd) (HSMul.hSMul x.fst x.snd)
    ⊢ Eq (tsum fun n => (NormedSpace.expSeries 𝕜 (TrivSqZeroExt R M) n) fun x_1 => …
  -/
  by_cases h : Summable (fun (n : ℕ) => (expSeries 𝕜 R n) fun _ ↦ fst x)
    /-
      case pos
      𝕜 : Type u_1
      R : Type u_3
      M : Type u_4
      inst✝¹⁸ : Field 𝕜
      inst✝¹⁷ : CharZero 𝕜
      inst✝¹⁶ : Ring R
      inst✝¹⁵ : AddCommGroup M
      inst✝¹⁴ : Algebra 𝕜 R
      inst✝¹³ : Module 𝕜 M
      inst✝¹² : Module R M
      inst✝¹¹ : Module (MulOpposite R) M
      inst✝¹⁰ : SMulCommClass R (MulOpposite R) M
      inst✝⁹ : IsScalarTower 𝕜 R M
      inst✝⁸ : IsScalarTower 𝕜 (MulOpposite R) M
      inst✝⁷ : TopologicalSpace R
      inst✝⁶ : TopologicalSpace M
      inst✝⁵ : TopologicalRing R
      inst✝⁴ : TopologicalAddGroup M
      inst✝³ : ContinuousSMul R M
      inst✝² : ContinuousSMul (MulOpposite R) M
      inst✝¹ : T2Space R
      inst✝ : T2Space M
      x : TrivSqZeroExt R M
      hx : Eq (HSMul.hSMul (MulOpposite.op x.fst) x.snd) (HSMul.hSMul x.fst x.snd)
      h : Summable fun n => (NormedSpace.expSeries 𝕜 R n) fun x_1 => x.fst
      ⊢ Eq (tsum fun n => (NormedSpace.expSeries 𝕜 (TrivSqZeroExt R M) n) fun x_1 => …
    -/
  · refine (hasSum_expSeries_of_smul_comm 𝕜 x hx ?_).tsum_eq
    /-
      case pos
      𝕜 : Type u_1
      R : Type u_3
      M : Type u_4
      inst✝¹⁸ : Field 𝕜
      inst✝¹⁷ : CharZero 𝕜
      inst✝¹⁶ : Ring R
      inst✝¹⁵ : AddCommGroup M
      inst✝¹⁴ : Algebra 𝕜 R
      inst✝¹³ : Module 𝕜 M
      inst✝¹² : Module R M
      inst✝¹¹ : Module (MulOpposite R) M
      inst✝¹⁰ : SMulCommClass R (MulOpposite R) M
      inst✝⁹ : IsScalarTower 𝕜 R M
      inst✝⁸ : IsScalarTower 𝕜 (MulOpposite R) M
      inst✝⁷ : TopologicalSpace R
      inst✝⁶ : TopologicalSpace M
      inst✝⁵ : TopologicalRing R
      inst✝⁴ : TopologicalAddGroup M
      inst✝³ : ContinuousSMul R M
      inst✝² : ContinuousSMul (MulOpposite R) M
      inst✝¹ : T2Space R
      inst✝ : T2Space M
      x : TrivSqZeroExt R M
      hx : Eq (HSMul.hSMul (MulOpposite.op x.fst) x.snd) (HSMul.hSMul x.fst x.snd)
      h : Summable fun n => (NormedSpace.expSeries 𝕜 R n) fun x_1 => x.fst
      ⊢ HasSum (fun n => (NormedSpace.expSeries 𝕜 R n) fun x_1 => x.fst) (tsum fun n …
    -/
    exact h.hasSum
    /-
      🎉 no goals
    -/
  · rw [tsum_eq_zero_of_not_summable h, zero_smul, inr_zero, inl_zero, zero_add,
      tsum_eq_zero_of_not_summable]
    /-
      case neg
      𝕜 : Type u_1
      R : Type u_3
      M : Type u_4
      inst✝¹⁸ : Field 𝕜
      inst✝¹⁷ : CharZero 𝕜
      inst✝¹⁶ : Ring R
      inst✝¹⁵ : AddCommGroup M
      inst✝¹⁴ : Algebra 𝕜 R
      inst✝¹³ : Module 𝕜 M
      inst✝¹² : Module R M
      inst✝¹¹ : Module (MulOpposite R) M
      inst✝¹⁰ : SMulCommClass R (MulOpposite R) M
      inst✝⁹ : IsScalarTower 𝕜 R M
      inst✝⁸ : IsScalarTower 𝕜 (MulOpposite R) M
      inst✝⁷ : TopologicalSpace R
      inst✝⁶ : TopologicalSpace M
      inst✝⁵ : TopologicalRing R
      inst✝⁴ : TopologicalAddGroup M
      inst✝³ : ContinuousSMul R M
      inst✝² : ContinuousSMul (MulOpposite R) M
      inst✝¹ : T2Space R
      inst✝ : T2Space M
      x : TrivSqZeroExt R M
      hx : Eq (HSMul.hSMul (MulOpposite.op x.fst) x.snd) (HSMul.hSMul x.fst x.snd)
      h : Not (Summable fun n => (NormedSpace.expSeries 𝕜 R n) fun x_1 => x.fst)
      ⊢ Not (Summable fun n => (NormedSpace.expSeries 𝕜 (TrivSqZeroExt R M) n) fun x …
    -/
    simp_rw [← fst_expSeries] at h
    /-
      case neg
      𝕜 : Type u_1
      R : Type u_3
      M : Type u_4
      inst✝¹⁸ : Field 𝕜
      inst✝¹⁷ : CharZero 𝕜
      inst✝¹⁶ : Ring R
      inst✝¹⁵ : AddCommGroup M
      inst✝¹⁴ : Algebra 𝕜 R
      inst✝¹³ : Module 𝕜 M
      inst✝¹² : Module R M
      inst✝¹¹ : Module (MulOpposite R) M
      inst✝¹⁰ : SMulCommClass R (MulOpposite R) M
      inst✝⁹ : IsScalarTower 𝕜 R M
      inst✝⁸ : IsScalarTower 𝕜 (MulOpposite R) M
      inst✝⁷ : TopologicalSpace R
      inst✝⁶ : TopologicalSpace M
      inst✝⁵ : TopologicalRing R
      inst✝⁴ : TopologicalAddGroup M
      inst✝³ : ContinuousSMul R M
      inst✝² : ContinuousSMul (MulOpposite R) M
      inst✝¹ : T2Space R
      inst✝ : T2Space M
      x : TrivSqZeroExt R M
      hx : Eq (HSMul.hSMul (MulOpposite.op x.fst) x.snd) (HSMul.hSMul x.fst x.snd)
      h : Not (Summable fun n => ((NormedSpace.expSeries 𝕜 (TrivSqZeroExt R M) n) fu …
      ⊢ Not (Summable fun n => (NormedSpace.expSeries 𝕜 (TrivSqZeroExt R M) n) fun x …
    -/
    refine mt ?_ h
    /-
      case neg
      𝕜 : Type u_1
      R : Type u_3
      M : Type u_4
      inst✝¹⁸ : Field 𝕜
      inst✝¹⁷ : CharZero 𝕜
      inst✝¹⁶ : Ring R
      inst✝¹⁵ : AddCommGroup M
      inst✝¹⁴ : Algebra 𝕜 R
      inst✝¹³ : Module 𝕜 M
      inst✝¹² : Module R M
      inst✝¹¹ : Module (MulOpposite R) M
      inst✝¹⁰ : SMulCommClass R (MulOpposite R) M
      inst✝⁹ : IsScalarTower 𝕜 R M
      inst✝⁸ : IsScalarTower 𝕜 (MulOpposite R) M
      inst✝⁷ : TopologicalSpace R
      inst✝⁶ : TopologicalSpace M
      inst✝⁵ : TopologicalRing R
      inst✝⁴ : TopologicalAddGroup M
      inst✝³ : ContinuousSMul R M
      inst✝² : ContinuousSMul (MulOpposite R) M
      inst✝¹ : T2Space R
      inst✝ : T2Space M
      x : TrivSqZeroExt R M
      hx : Eq (HSMul.hSMul (MulOpposite.op x.fst) x.snd) (HSMul.hSMul x.fst x.snd)
      h : Not (Summable fun n => ((NormedSpace.expSeries 𝕜 (TrivSqZeroExt R M) n) fu …
      ⊢ (Summable fun n => (NormedSpace.expSeries 𝕜 (TrivSqZeroExt R M) n) fun x_1 = …
    -/
    exact (Summable.map · (TrivSqZeroExt.fstHom 𝕜 R M).toLinearMap continuous_fst)
    /-
      🎉 no goals
    -/


@[simp]
theorem exp_inl (x : R) : exp 𝕜 (inl x : tsze R M) = inl (exp 𝕜 x) := by
  /-
    𝕜 : Type u_1
    R : Type u_3
    M : Type u_4
    inst✝¹⁸ : Field 𝕜
    inst✝¹⁷ : CharZero 𝕜
    inst✝¹⁶ : Ring R
    inst✝¹⁵ : AddCommGroup M
    inst✝¹⁴ : Algebra 𝕜 R
    inst✝¹³ : Module 𝕜 M
    inst✝¹² : Module R M
    inst✝¹¹ : Module (MulOpposite R) M
    inst✝¹⁰ : SMulCommClass R (MulOpposite R) M
    inst✝⁹ : IsScalarTower 𝕜 R M
    inst✝⁸ : IsScalarTower 𝕜 (MulOpposite R) M
    inst✝⁷ : TopologicalSpace R
    inst✝⁶ : TopologicalSpace M
    inst✝⁵ : TopologicalRing R
    inst✝⁴ : TopologicalAddGroup M
    inst✝³ : ContinuousSMul R M
    inst✝² : ContinuousSMul (MulOpposite R) M
    inst✝¹ : T2Space R
    inst✝ : T2Space M
    x : R
    ⊢ Eq (NormedSpace.exp 𝕜 (TrivSqZeroExt.inl x)) (TrivSqZeroExt.inl (NormedSpace …
  -/
  rw [exp_def_of_smul_comm, snd_inl, fst_inl, smul_zero, inr_zero, add_zero]
  /-
    case hx
    𝕜 : Type u_1
    R : Type u_3
    M : Type u_4
    inst✝¹⁸ : Field 𝕜
    inst✝¹⁷ : CharZero 𝕜
    inst✝¹⁶ : Ring R
    inst✝¹⁵ : AddCommGroup M
    inst✝¹⁴ : Algebra 𝕜 R
    inst✝¹³ : Module 𝕜 M
    inst✝¹² : Module R M
    inst✝¹¹ : Module (MulOpposite R) M
    inst✝¹⁰ : SMulCommClass R (MulOpposite R) M
    inst✝⁹ : IsScalarTower 𝕜 R M
    inst✝⁸ : IsScalarTower 𝕜 (MulOpposite R) M
    inst✝⁷ : TopologicalSpace R
    inst✝⁶ : TopologicalSpace M
    inst✝⁵ : TopologicalRing R
    inst✝⁴ : TopologicalAddGroup M
    inst✝³ : ContinuousSMul R M
    inst✝² : ContinuousSMul (MulOpposite R) M
    inst✝¹ : T2Space R
    inst✝ : T2Space M
    x : R
    ⊢ Eq (HSMul.hSMul (MulOpposite.op (TrivSqZeroExt.inl x).fst) (TrivSqZeroExt.in …
  -/
  rw [snd_inl, fst_inl, smul_zero, smul_zero]
  /-
    🎉 no goals
  -/


@[simp]
theorem exp_inr (m : M) : exp 𝕜 (inr m : tsze R M) = 1 + inr m := by
  /-
    𝕜 : Type u_1
    R : Type u_3
    M : Type u_4
    inst✝¹⁸ : Field 𝕜
    inst✝¹⁷ : CharZero 𝕜
    inst✝¹⁶ : Ring R
    inst✝¹⁵ : AddCommGroup M
    inst✝¹⁴ : Algebra 𝕜 R
    inst✝¹³ : Module 𝕜 M
    inst✝¹² : Module R M
    inst✝¹¹ : Module (MulOpposite R) M
    inst✝¹⁰ : SMulCommClass R (MulOpposite R) M
    inst✝⁹ : IsScalarTower 𝕜 R M
    inst✝⁸ : IsScalarTower 𝕜 (MulOpposite R) M
    inst✝⁷ : TopologicalSpace R
    inst✝⁶ : TopologicalSpace M
    inst✝⁵ : TopologicalRing R
    inst✝⁴ : TopologicalAddGroup M
    inst✝³ : ContinuousSMul R M
    inst✝² : ContinuousSMul (MulOpposite R) M
    inst✝¹ : T2Space R
    inst✝ : T2Space M
    m : M
    ⊢ Eq (NormedSpace.exp 𝕜 (TrivSqZeroExt.inr m)) (HAdd.hAdd 1 (TrivSqZeroExt.inr …
  -/
  rw [exp_def_of_smul_comm, snd_inr, fst_inr, exp_zero, one_smul, inl_one]
  /-
    case hx
    𝕜 : Type u_1
    R : Type u_3
    M : Type u_4
    inst✝¹⁸ : Field 𝕜
    inst✝¹⁷ : CharZero 𝕜
    inst✝¹⁶ : Ring R
    inst✝¹⁵ : AddCommGroup M
    inst✝¹⁴ : Algebra 𝕜 R
    inst✝¹³ : Module 𝕜 M
    inst✝¹² : Module R M
    inst✝¹¹ : Module (MulOpposite R) M
    inst✝¹⁰ : SMulCommClass R (MulOpposite R) M
    inst✝⁹ : IsScalarTower 𝕜 R M
    inst✝⁸ : IsScalarTower 𝕜 (MulOpposite R) M
    inst✝⁷ : TopologicalSpace R
    inst✝⁶ : TopologicalSpace M
    inst✝⁵ : TopologicalRing R
    inst✝⁴ : TopologicalAddGroup M
    inst✝³ : ContinuousSMul R M
    inst✝² : ContinuousSMul (MulOpposite R) M
    inst✝¹ : T2Space R
    inst✝ : T2Space M
    m : M
    ⊢ Eq (HSMul.hSMul (MulOpposite.op (TrivSqZeroExt.inr m).fst) (TrivSqZeroExt.in …
  -/
  rw [snd_inr, fst_inr, MulOpposite.op_zero, zero_smul, zero_smul]
  /-
    🎉 no goals
  -/


theorem exp_def (x : tsze R M) : exp 𝕜 x = inl (exp 𝕜 x.fst) + inr (exp 𝕜 x.fst • x.snd) :=
  exp_def_of_smul_comm 𝕜 x (op_smul_eq_smul _ _)


@[simp]
theorem fst_exp (x : tsze R M) : fst (exp 𝕜 x) = exp 𝕜 x.fst := by
  /-
    𝕜 : Type u_1
    R : Type u_3
    M : Type u_4
    inst✝¹⁷ : Field 𝕜
    inst✝¹⁶ : CharZero 𝕜
    inst✝¹⁵ : CommRing R
    inst✝¹⁴ : AddCommGroup M
    inst✝¹³ : Algebra 𝕜 R
    inst✝¹² : Module 𝕜 M
    inst✝¹¹ : Module R M
    inst✝¹⁰ : Module (MulOpposite R) M
    inst✝⁹ : IsCentralScalar R M
    inst✝⁸ : IsScalarTower 𝕜 R M
    inst✝⁷ : TopologicalSpace R
    inst✝⁶ : TopologicalSpace M
    inst✝⁵ : TopologicalRing R
    inst✝⁴ : TopologicalAddGroup M
    inst✝³ : ContinuousSMul R M
    inst✝² : ContinuousSMul (MulOpposite R) M
    inst✝¹ : T2Space R
    inst✝ : T2Space M
    x : TrivSqZeroExt R M
    ⊢ Eq (NormedSpace.exp 𝕜 x).fst (NormedSpace.exp 𝕜 x.fst)
  -/
  rw [exp_def, fst_add, fst_inl, fst_inr, add_zero]
  /-
    🎉 no goals
  -/


@[simp]
theorem snd_exp (x : tsze R M) : snd (exp 𝕜 x) = exp 𝕜 x.fst • x.snd := by
  /-
    𝕜 : Type u_1
    R : Type u_3
    M : Type u_4
    inst✝¹⁷ : Field 𝕜
    inst✝¹⁶ : CharZero 𝕜
    inst✝¹⁵ : CommRing R
    inst✝¹⁴ : AddCommGroup M
    inst✝¹³ : Algebra 𝕜 R
    inst✝¹² : Module 𝕜 M
    inst✝¹¹ : Module R M
    inst✝¹⁰ : Module (MulOpposite R) M
    inst✝⁹ : IsCentralScalar R M
    inst✝⁸ : IsScalarTower 𝕜 R M
    inst✝⁷ : TopologicalSpace R
    inst✝⁶ : TopologicalSpace M
    inst✝⁵ : TopologicalRing R
    inst✝⁴ : TopologicalAddGroup M
    inst✝³ : ContinuousSMul R M
    inst✝² : ContinuousSMul (MulOpposite R) M
    inst✝¹ : T2Space R
    inst✝ : T2Space M
    x : TrivSqZeroExt R M
    ⊢ Eq (NormedSpace.exp 𝕜 x).snd (HSMul.hSMul (NormedSpace.exp 𝕜 x.fst) x.snd)
  -/
  rw [exp_def, snd_add, snd_inl, snd_inr, zero_add]
  /-
    🎉 no goals
  -/


/-- Polar form of trivial-square-zero extension. -/
theorem eq_smul_exp_of_invertible (x : tsze R M) [Invertible x.fst] :
    x = x.fst • exp 𝕜 (⅟ x.fst • inr x.snd) := by
  rw [← inr_smul, exp_inr, smul_add, ← inl_one, ← inl_smul, ← inr_smul, smul_eq_mul, mul_one,
    smul_smul, mul_invOf_self, one_smul, inl_fst_add_inr_snd_eq]


/-- More convenient version of `TrivSqZeroExt.eq_smul_exp_of_invertible` for when `R` is a
field. -/
theorem eq_smul_exp_of_ne_zero (x : tsze R M) (hx : x.fst ≠ 0) :
    x = x.fst • exp 𝕜 (x.fst⁻¹ • inr x.snd) :=
  letI : Invertible x.fst := invertibleOfNonzero hx
  eq_smul_exp_of_invertible _ _


instance instL1SeminormedAddCommGroup : SeminormedAddCommGroup (tsze R M) :=
  inferInstanceAs <| SeminormedAddCommGroup (WithLp 1 <| R × M)


theorem norm_def (x : tsze R M) : ‖x‖ = ‖fst x‖ + ‖snd x‖ := by
  /-
    R : Type u_3
    M : Type u_4
    inst✝¹ : SeminormedRing R
    inst✝ : SeminormedAddCommGroup M
    x : TrivSqZeroExt R M
    ⊢ Eq (Norm.norm x) (HAdd.hAdd (Norm.norm x.fst) (Norm.norm x.snd))
  -/
  rw [WithLp.prod_norm_eq_add (by norm_num)]
  /-
    R : Type u_3
    M : Type u_4
    inst✝¹ : SeminormedRing R
    inst✝ : SeminormedAddCommGroup M
    x : TrivSqZeroExt R M
    ⊢ Eq (HPow.hPow (HAdd.hAdd (HPow.hPow (Norm.norm x.1) (ENNReal.toReal 1)) (HPo …
  -/
  simp only [ENNReal.one_toReal, Real.rpow_one, div_one]
  /-
    R : Type u_3
    M : Type u_4
    inst✝¹ : SeminormedRing R
    inst✝ : SeminormedAddCommGroup M
    x : TrivSqZeroExt R M
    ⊢ Eq (HAdd.hAdd (Norm.norm x.1) (Norm.norm x.2)) (HAdd.hAdd (Norm.norm x.fst)  …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem nnnorm_def (x : tsze R M) : ‖x‖₊ = ‖fst x‖₊ + ‖snd x‖₊ := by
  /-
    R : Type u_3
    M : Type u_4
    inst✝¹ : SeminormedRing R
    inst✝ : SeminormedAddCommGroup M
    x : TrivSqZeroExt R M
    ⊢ Eq (NNNorm.nnnorm x) (HAdd.hAdd (NNNorm.nnnorm x.fst) (NNNorm.nnnorm x.snd))
  -/
  ext; simp [norm_def]
       /-
         🎉 no goals
       -/


                                                                    /-
                                                                      R : Type u_3
                                                                      M : Type u_4
                                                                      inst✝¹ : SeminormedRing R
                                                                      inst✝ : SeminormedAddCommGroup M
                                                                      r : R
                                                                      ⊢ Eq (Norm.norm (TrivSqZeroExt.inl r)) (Norm.norm r)
                                                                    -/
@[simp] theorem norm_inl (r : R) : ‖(inl r : tsze R M)‖ = ‖r‖ := by simp [norm_def]
                                                                    /-
                                                                      🎉 no goals
                                                                    -/

                                                                    /-
                                                                      R : Type u_3
                                                                      M : Type u_4
                                                                      inst✝¹ : SeminormedRing R
                                                                      inst✝ : SeminormedAddCommGroup M
                                                                      m : M
                                                                      ⊢ Eq (Norm.norm (TrivSqZeroExt.inr m)) (Norm.norm m)
                                                                    -/
@[simp] theorem norm_inr (m : M) : ‖(inr m : tsze R M)‖ = ‖m‖ := by simp [norm_def]
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


                                                                        /-
                                                                          R : Type u_3
                                                                          M : Type u_4
                                                                          inst✝¹ : SeminormedRing R
                                                                          inst✝ : SeminormedAddCommGroup M
                                                                          r : R
                                                                          ⊢ Eq (NNNorm.nnnorm (TrivSqZeroExt.inl r)) (NNNorm.nnnorm r)
                                                                        -/
@[simp] theorem nnnorm_inl (r : R) : ‖(inl r : tsze R M)‖₊ = ‖r‖₊ := by simp [nnnorm_def]
                                                                        /-
                                                                          🎉 no goals
                                                                        -/

                                                                        /-
                                                                          R : Type u_3
                                                                          M : Type u_4
                                                                          inst✝¹ : SeminormedRing R
                                                                          inst✝ : SeminormedAddCommGroup M
                                                                          m : M
                                                                          ⊢ Eq (NNNorm.nnnorm (TrivSqZeroExt.inr m)) (NNNorm.nnnorm m)
                                                                        -/
@[simp] theorem nnnorm_inr (m : M) : ‖(inr m : tsze R M)‖₊ = ‖m‖₊ := by simp [nnnorm_def]
                                                                        /-
                                                                          🎉 no goals
                                                                        -/


instance instL1SeminormedRing : SeminormedRing (tsze R M) where
  norm_mul
  | ⟨r₁, m₁⟩, ⟨r₂, m₂⟩ => by
    /-
      𝕜 : Type u_1
      S : Type u_2
      R : Type u_3
      M : Type u_4
      inst✝¹¹ : SeminormedCommRing S
      inst✝¹⁰ : SeminormedRing R
      inst✝⁹ : SeminormedAddCommGroup M
      inst✝⁸ : Algebra S R
      inst✝⁷ : Module S M
      inst✝⁶ : BoundedSMul S R
      inst✝⁵ : BoundedSMul S M
      inst✝⁴ : Module R M
      inst✝³ : BoundedSMul R M
      inst✝² : Module (MulOpposite R) M
      inst✝¹ : BoundedSMul (MulOpposite R) M
      inst✝ : SMulCommClass R (MulOpposite R) M
      r₁ : R
      m₁ : M
      r₂ : R
      m₂ : M
      ⊢ LE.le (Norm.norm (HMul.hMul { fst := r₁, snd := m₁ } { fst := r₂, snd := m₂  …
    -/
    dsimp
    /-
      𝕜 : Type u_1
      S : Type u_2
      R : Type u_3
      M : Type u_4
      inst✝¹¹ : SeminormedCommRing S
      inst✝¹⁰ : SeminormedRing R
      inst✝⁹ : SeminormedAddCommGroup M
      inst✝⁸ : Algebra S R
      inst✝⁷ : Module S M
      inst✝⁶ : BoundedSMul S R
      inst✝⁵ : BoundedSMul S M
      inst✝⁴ : Module R M
      inst✝³ : BoundedSMul R M
      inst✝² : Module (MulOpposite R) M
      inst✝¹ : BoundedSMul (MulOpposite R) M
      inst✝ : SMulCommClass R (MulOpposite R) M
      r₁ : R
      m₁ : M
      r₂ : R
      m₂ : M
      ⊢ LE.le (Norm.norm (HMul.hMul { fst := r₁, snd := m₁ } { fst := r₂, snd := m₂  …
    -/
    rw [norm_def, norm_def, norm_def, add_mul, mul_add, mul_add, snd_mul, fst_mul]
    /-
      𝕜 : Type u_1
      S : Type u_2
      R : Type u_3
      M : Type u_4
      inst✝¹¹ : SeminormedCommRing S
      inst✝¹⁰ : SeminormedRing R
      inst✝⁹ : SeminormedAddCommGroup M
      inst✝⁸ : Algebra S R
      inst✝⁷ : Module S M
      inst✝⁶ : BoundedSMul S R
      inst✝⁵ : BoundedSMul S M
      inst✝⁴ : Module R M
      inst✝³ : BoundedSMul R M
      inst✝² : Module (MulOpposite R) M
      inst✝¹ : BoundedSMul (MulOpposite R) M
      inst✝ : SMulCommClass R (MulOpposite R) M
      r₁ : R
      m₁ : M
      r₂ : R
      m₂ : M
      ⊢ LE.le (HAdd.hAdd (Norm.norm (HMul.hMul (TrivSqZeroExt.fst { fst := r₁, snd : …
    -/
    dsimp [fst, snd]
    /-
      𝕜 : Type u_1
      S : Type u_2
      R : Type u_3
      M : Type u_4
      inst✝¹¹ : SeminormedCommRing S
      inst✝¹⁰ : SeminormedRing R
      inst✝⁹ : SeminormedAddCommGroup M
      inst✝⁸ : Algebra S R
      inst✝⁷ : Module S M
      inst✝⁶ : BoundedSMul S R
      inst✝⁵ : BoundedSMul S M
      inst✝⁴ : Module R M
      inst✝³ : BoundedSMul R M
      inst✝² : Module (MulOpposite R) M
      inst✝¹ : BoundedSMul (MulOpposite R) M
      inst✝ : SMulCommClass R (MulOpposite R) M
      r₁ : R
      m₁ : M
      r₂ : R
      m₂ : M
      ⊢ LE.le (HAdd.hAdd (Norm.norm (HMul.hMul r₁ r₂)) (Norm.norm (HAdd.hAdd (HSMul. …
    -/
    rw [add_assoc]
    /-
      𝕜 : Type u_1
      S : Type u_2
      R : Type u_3
      M : Type u_4
      inst✝¹¹ : SeminormedCommRing S
      inst✝¹⁰ : SeminormedRing R
      inst✝⁹ : SeminormedAddCommGroup M
      inst✝⁸ : Algebra S R
      inst✝⁷ : Module S M
      inst✝⁶ : BoundedSMul S R
      inst✝⁵ : BoundedSMul S M
      inst✝⁴ : Module R M
      inst✝³ : BoundedSMul R M
      inst✝² : Module (MulOpposite R) M
      inst✝¹ : BoundedSMul (MulOpposite R) M
      inst✝ : SMulCommClass R (MulOpposite R) M
      r₁ : R
      m₁ : M
      r₂ : R
      m₂ : M
      ⊢ LE.le (HAdd.hAdd (Norm.norm (HMul.hMul r₁ r₂)) (Norm.norm (HAdd.hAdd (HSMul. …
    -/
    gcongr
      /-
        case h₁
        𝕜 : Type u_1
        S : Type u_2
        R : Type u_3
        M : Type u_4
        inst✝¹¹ : SeminormedCommRing S
        inst✝¹⁰ : SeminormedRing R
        inst✝⁹ : SeminormedAddCommGroup M
        inst✝⁸ : Algebra S R
        inst✝⁷ : Module S M
        inst✝⁶ : BoundedSMul S R
        inst✝⁵ : BoundedSMul S M
        inst✝⁴ : Module R M
        inst✝³ : BoundedSMul R M
        inst✝² : Module (MulOpposite R) M
        inst✝¹ : BoundedSMul (MulOpposite R) M
        inst✝ : SMulCommClass R (MulOpposite R) M
        r₁ : R
        m₁ : M
        r₂ : R
        m₂ : M
        ⊢ LE.le (Norm.norm (HMul.hMul r₁ r₂)) (HMul.hMul (Norm.norm r₁) (Norm.norm r₂))
      -/
    · exact norm_mul_le _ _
      /-
        🎉 no goals
      -/
    /-
      case h₂
      𝕜 : Type u_1
      S : Type u_2
      R : Type u_3
      M : Type u_4
      inst✝¹¹ : SeminormedCommRing S
      inst✝¹⁰ : SeminormedRing R
      inst✝⁹ : SeminormedAddCommGroup M
      inst✝⁸ : Algebra S R
      inst✝⁷ : Module S M
      inst✝⁶ : BoundedSMul S R
      inst✝⁵ : BoundedSMul S M
      inst✝⁴ : Module R M
      inst✝³ : BoundedSMul R M
      inst✝² : Module (MulOpposite R) M
      inst✝¹ : BoundedSMul (MulOpposite R) M
      inst✝ : SMulCommClass R (MulOpposite R) M
      r₁ : R
      m₁ : M
      r₂ : R
      m₂ : M
      ⊢ LE.le (Norm.norm (HAdd.hAdd (HSMul.hSMul r₁ m₂) (HSMul.hSMul (MulOpposite.op …
    -/
    refine (norm_add_le _ _).trans ?_
    /-
      case h₂
      𝕜 : Type u_1
      S : Type u_2
      R : Type u_3
      M : Type u_4
      inst✝¹¹ : SeminormedCommRing S
      inst✝¹⁰ : SeminormedRing R
      inst✝⁹ : SeminormedAddCommGroup M
      inst✝⁸ : Algebra S R
      inst✝⁷ : Module S M
      inst✝⁶ : BoundedSMul S R
      inst✝⁵ : BoundedSMul S M
      inst✝⁴ : Module R M
      inst✝³ : BoundedSMul R M
      inst✝² : Module (MulOpposite R) M
      inst✝¹ : BoundedSMul (MulOpposite R) M
      inst✝ : SMulCommClass R (MulOpposite R) M
      r₁ : R
      m₁ : M
      r₂ : R
      m₂ : M
      ⊢ LE.le (HAdd.hAdd (Norm.norm (HSMul.hSMul r₁ m₂)) (Norm.norm (HSMul.hSMul (Mu …
    -/
    gcongr
      /-
        case h₂.h₁
        𝕜 : Type u_1
        S : Type u_2
        R : Type u_3
        M : Type u_4
        inst✝¹¹ : SeminormedCommRing S
        inst✝¹⁰ : SeminormedRing R
        inst✝⁹ : SeminormedAddCommGroup M
        inst✝⁸ : Algebra S R
        inst✝⁷ : Module S M
        inst✝⁶ : BoundedSMul S R
        inst✝⁵ : BoundedSMul S M
        inst✝⁴ : Module R M
        inst✝³ : BoundedSMul R M
        inst✝² : Module (MulOpposite R) M
        inst✝¹ : BoundedSMul (MulOpposite R) M
        inst✝ : SMulCommClass R (MulOpposite R) M
        r₁ : R
        m₁ : M
        r₂ : R
        m₂ : M
        ⊢ LE.le (Norm.norm (HSMul.hSMul r₁ m₂)) (HMul.hMul (Norm.norm r₁) (Norm.norm m …
      -/
    · exact norm_smul_le _ _
      /-
        🎉 no goals
      -/
    /-
      case h₂.h₂
      𝕜 : Type u_1
      S : Type u_2
      R : Type u_3
      M : Type u_4
      inst✝¹¹ : SeminormedCommRing S
      inst✝¹⁰ : SeminormedRing R
      inst✝⁹ : SeminormedAddCommGroup M
      inst✝⁸ : Algebra S R
      inst✝⁷ : Module S M
      inst✝⁶ : BoundedSMul S R
      inst✝⁵ : BoundedSMul S M
      inst✝⁴ : Module R M
      inst✝³ : BoundedSMul R M
      inst✝² : Module (MulOpposite R) M
      inst✝¹ : BoundedSMul (MulOpposite R) M
      inst✝ : SMulCommClass R (MulOpposite R) M
      r₁ : R
      m₁ : M
      r₂ : R
      m₂ : M
      ⊢ LE.le (Norm.norm (HSMul.hSMul (MulOpposite.op r₂) m₁)) (HAdd.hAdd (HMul.hMul …
    -/
    refine (_root_.norm_smul_le _ _).trans ?_
    /-
      case h₂.h₂
      𝕜 : Type u_1
      S : Type u_2
      R : Type u_3
      M : Type u_4
      inst✝¹¹ : SeminormedCommRing S
      inst✝¹⁰ : SeminormedRing R
      inst✝⁹ : SeminormedAddCommGroup M
      inst✝⁸ : Algebra S R
      inst✝⁷ : Module S M
      inst✝⁶ : BoundedSMul S R
      inst✝⁵ : BoundedSMul S M
      inst✝⁴ : Module R M
      inst✝³ : BoundedSMul R M
      inst✝² : Module (MulOpposite R) M
      inst✝¹ : BoundedSMul (MulOpposite R) M
      inst✝ : SMulCommClass R (MulOpposite R) M
      r₁ : R
      m₁ : M
      r₂ : R
      m₂ : M
      ⊢ LE.le (HMul.hMul (Norm.norm (MulOpposite.op r₂)) (Norm.norm m₁)) (HAdd.hAdd  …
    -/
    rw [mul_comm, MulOpposite.norm_op]
    /-
      case h₂.h₂
      𝕜 : Type u_1
      S : Type u_2
      R : Type u_3
      M : Type u_4
      inst✝¹¹ : SeminormedCommRing S
      inst✝¹⁰ : SeminormedRing R
      inst✝⁹ : SeminormedAddCommGroup M
      inst✝⁸ : Algebra S R
      inst✝⁷ : Module S M
      inst✝⁶ : BoundedSMul S R
      inst✝⁵ : BoundedSMul S M
      inst✝⁴ : Module R M
      inst✝³ : BoundedSMul R M
      inst✝² : Module (MulOpposite R) M
      inst✝¹ : BoundedSMul (MulOpposite R) M
      inst✝ : SMulCommClass R (MulOpposite R) M
      r₁ : R
      m₁ : M
      r₂ : R
      m₂ : M
      ⊢ LE.le (HMul.hMul (Norm.norm m₁) (Norm.norm r₂)) (HAdd.hAdd (HMul.hMul (Norm. …
    -/
    exact le_add_of_nonneg_right <| by positivity
    /-
      🎉 no goals
    -/
  __ : SeminormedAddCommGroup (tsze R M) := inferInstance
  __ : Ring (tsze R M) := inferInstance


instance instL1BoundedSMul : BoundedSMul S (tsze R M) :=
  inferInstanceAs <| BoundedSMul S (WithLp 1 <| R × M)


instance [NormOneClass R] : NormOneClass (tsze R M) where
                 /-
                   𝕜 : Type u_1
                   S : Type u_2
                   R : Type u_3
                   M : Type u_4
                   inst✝¹² : SeminormedCommRing S
                   inst✝¹¹ : SeminormedRing R
                   inst✝¹⁰ : SeminormedAddCommGroup M
                   inst✝⁹ : Algebra S R
                   inst✝⁸ : Module S M
                   inst✝⁷ : BoundedSMul S R
                   inst✝⁶ : BoundedSMul S M
                   inst✝⁵ : Module R M
                   inst✝⁴ : BoundedSMul R M
                   inst✝³ : Module (MulOpposite R) M
                   inst✝² : BoundedSMul (MulOpposite R) M
                   inst✝¹ : SMulCommClass R (MulOpposite R) M
                   inst✝ : NormOneClass R
                   ⊢ Eq (Norm.norm 1) 1
                 -/
  norm_one := by rw [norm_def, fst_one, snd_one, norm_zero, norm_one, add_zero]
                 /-
                   🎉 no goals
                 -/



instance instL1SeminormedCommRing : SeminormedCommRing (tsze R M) where
  __ : CommRing (tsze R M) := inferInstance
  __ : SeminormedRing (tsze R M) := inferInstance


instance instL1NormedAddCommGroup : NormedAddCommGroup (tsze R M) :=
  inferInstanceAs <| NormedAddCommGroup (WithLp 1 <| R × M)


instance instL1NormedRing : NormedRing (tsze R M) where
  __ : NormedAddCommGroup (tsze R M) := inferInstance
  __ : SeminormedRing (tsze R M) := inferInstance


instance instL1NormedCommRing : NormedCommRing (tsze R M) where
  __ : CommRing (tsze R M) := inferInstance
  __ : NormedRing (tsze R M) := inferInstance


instance instL1NormedSpace : NormedSpace 𝕜 (tsze R M) :=
  inferInstanceAs <| NormedSpace 𝕜 (WithLp 1 <| R × M)


instance instL1NormedAlgebra : NormedAlgebra 𝕜 (tsze R M) where
  norm_smul_le := _root_.norm_smul_le


