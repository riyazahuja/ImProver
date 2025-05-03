theorem HasSum.mul_left (a₂) (h : HasSum f a₁) : HasSum (fun i ↦ a₂ * f i) (a₂ * a₁) := by
  /-
    ι : Type u_1
    α : Type u_3
    inst✝² : NonUnitalNonAssocSemiring α
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSemiring α
    f : ι → α
    a₁ a₂ : α
    h : HasSum f a₁
    ⊢ HasSum (fun i => HMul.hMul a₂ (f i)) (HMul.hMul a₂ a₁)
  -/
  simpa only using h.map (AddMonoidHom.mulLeft a₂) (continuous_const.mul continuous_id)
  /-
    🎉 no goals
  -/


theorem HasSum.mul_right (a₂) (hf : HasSum f a₁) : HasSum (fun i ↦ f i * a₂) (a₁ * a₂) := by
  /-
    ι : Type u_1
    α : Type u_3
    inst✝² : NonUnitalNonAssocSemiring α
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSemiring α
    f : ι → α
    a₁ a₂ : α
    hf : HasSum f a₁
    ⊢ HasSum (fun i => HMul.hMul (f i) a₂) (HMul.hMul a₁ a₂)
  -/
  simpa only using hf.map (AddMonoidHom.mulRight a₂) (continuous_id.mul continuous_const)
  /-
    🎉 no goals
  -/


theorem Summable.mul_left (a) (hf : Summable f) : Summable fun i ↦ a * f i :=
  (hf.hasSum.mul_left _).summable


theorem Summable.mul_right (a) (hf : Summable f) : Summable fun i ↦ f i * a :=
  (hf.hasSum.mul_right _).summable


theorem Summable.tsum_mul_left (a) (hf : Summable f) : ∑' i, a * f i = a * ∑' i, f i :=
  (hf.hasSum.mul_left _).tsum_eq


theorem Summable.tsum_mul_right (a) (hf : Summable f) : ∑' i, f i * a = (∑' i, f i) * a :=
  (hf.hasSum.mul_right _).tsum_eq


theorem Commute.tsum_right (a) (h : ∀ i, Commute a (f i)) : Commute a (∑' i, f i) := by
  classical
  by_cases hf : Summable f
  · exact (hf.tsum_mul_left a).symm.trans ((congr_arg _ <| funext h).trans (hf.tsum_mul_right a))
  · exact (tsum_eq_zero_of_not_summable hf).symm ▸ Commute.zero_right _


theorem Commute.tsum_left (a) (h : ∀ i, Commute (f i) a) : Commute (∑' i, f i) a :=
  (Commute.tsum_right _ fun i ↦ (h i).symm).symm


theorem HasSum.div_const (h : HasSum f a) (b : α) : HasSum (fun i ↦ f i / b) (a / b) := by
  /-
    ι : Type u_1
    α : Type u_3
    inst✝² : DivisionSemiring α
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSemiring α
    f : ι → α
    a : α
    h : HasSum f a
    b : α
    ⊢ HasSum (fun i => HDiv.hDiv (f i) b) (HDiv.hDiv a b)
  -/
  simp only [div_eq_mul_inv, h.mul_right b⁻¹]
  /-
    🎉 no goals
  -/


theorem Summable.div_const (h : Summable f) (b : α) : Summable fun i ↦ f i / b :=
  (h.hasSum.div_const _).summable


theorem hasSum_mul_left_iff (h : a₂ ≠ 0) : HasSum (fun i ↦ a₂ * f i) (a₂ * a₁) ↔ HasSum f a₁ :=
              /-
                ι : Type u_1
                α : Type u_3
                inst✝² : DivisionSemiring α
                inst✝¹ : TopologicalSpace α
                inst✝ : TopologicalSemiring α
                f : ι → α
                a₁ a₂ : α
                h : Ne a₂ 0
                H : HasSum (fun i => HMul.hMul a₂ (f i)) (HMul.hMul a₂ a₁)
                ⊢ HasSum f a₁
              -/
  ⟨fun H ↦ by simpa only [inv_mul_cancel_left₀ h] using H.mul_left a₂⁻¹, HasSum.mul_left _⟩
              /-
                🎉 no goals
              -/


theorem hasSum_mul_right_iff (h : a₂ ≠ 0) : HasSum (fun i ↦ f i * a₂) (a₁ * a₂) ↔ HasSum f a₁ :=
              /-
                ι : Type u_1
                α : Type u_3
                inst✝² : DivisionSemiring α
                inst✝¹ : TopologicalSpace α
                inst✝ : TopologicalSemiring α
                f : ι → α
                a₁ a₂ : α
                h : Ne a₂ 0
                H : HasSum (fun i => HMul.hMul (f i) a₂) (HMul.hMul a₁ a₂)
                ⊢ HasSum f a₁
              -/
  ⟨fun H ↦ by simpa only [mul_inv_cancel_right₀ h] using H.mul_right a₂⁻¹, HasSum.mul_right _⟩
              /-
                🎉 no goals
              -/


theorem hasSum_div_const_iff (h : a₂ ≠ 0) : HasSum (fun i ↦ f i / a₂) (a₁ / a₂) ↔ HasSum f a₁ := by
  /-
    ι : Type u_1
    α : Type u_3
    inst✝² : DivisionSemiring α
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSemiring α
    f : ι → α
    a₁ a₂ : α
    h : Ne a₂ 0
    ⊢ Iff (HasSum (fun i => HDiv.hDiv (f i) a₂) (HDiv.hDiv a₁ a₂)) (HasSum f a₁)
  -/
  simpa only [div_eq_mul_inv] using hasSum_mul_right_iff (inv_ne_zero h)
  /-
    🎉 no goals
  -/


theorem summable_mul_left_iff (h : a ≠ 0) : (Summable fun i ↦ a * f i) ↔ Summable f :=
              /-
                ι : Type u_1
                α : Type u_3
                inst✝² : DivisionSemiring α
                inst✝¹ : TopologicalSpace α
                inst✝ : TopologicalSemiring α
                f : ι → α
                a : α
                h : Ne a 0
                H : Summable fun i => HMul.hMul a (f i)
                ⊢ Summable f
              -/
  ⟨fun H ↦ by simpa only [inv_mul_cancel_left₀ h] using H.mul_left a⁻¹, fun H ↦ H.mul_left _⟩
              /-
                🎉 no goals
              -/


theorem summable_mul_right_iff (h : a ≠ 0) : (Summable fun i ↦ f i * a) ↔ Summable f :=
              /-
                ι : Type u_1
                α : Type u_3
                inst✝² : DivisionSemiring α
                inst✝¹ : TopologicalSpace α
                inst✝ : TopologicalSemiring α
                f : ι → α
                a : α
                h : Ne a 0
                H : Summable fun i => HMul.hMul (f i) a
                ⊢ Summable f
              -/
  ⟨fun H ↦ by simpa only [mul_inv_cancel_right₀ h] using H.mul_right a⁻¹, fun H ↦ H.mul_right _⟩
              /-
                🎉 no goals
              -/


theorem summable_div_const_iff (h : a ≠ 0) : (Summable fun i ↦ f i / a) ↔ Summable f := by
  /-
    ι : Type u_1
    α : Type u_3
    inst✝² : DivisionSemiring α
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSemiring α
    f : ι → α
    a : α
    h : Ne a 0
    ⊢ Iff (Summable fun i => HDiv.hDiv (f i) a) (Summable f)
  -/
  simpa only [div_eq_mul_inv] using summable_mul_right_iff (inv_ne_zero h)
  /-
    🎉 no goals
  -/


theorem tsum_mul_left [T2Space α] : ∑' x, a * f x = a * ∑' x, f x := by
  classical
  exact if hf : Summable f then hf.tsum_mul_left a
  else if ha : a = 0 then by simp [ha]
  else by rw [tsum_eq_zero_of_not_summable hf,
              tsum_eq_zero_of_not_summable (mt (summable_mul_left_iff ha).mp hf), mul_zero]


theorem tsum_mul_right [T2Space α] : ∑' x, f x * a = (∑' x, f x) * a := by
  classical
  exact if hf : Summable f then hf.tsum_mul_right a
  else if ha : a = 0 then by simp [ha]
  else by rw [tsum_eq_zero_of_not_summable hf,
              tsum_eq_zero_of_not_summable (mt (summable_mul_right_iff ha).mp hf), zero_mul]


theorem tsum_div_const [T2Space α] : ∑' x, f x / a = (∑' x, f x) / a := by
  /-
    ι : Type u_1
    α : Type u_3
    inst✝³ : DivisionSemiring α
    inst✝² : TopologicalSpace α
    inst✝¹ : TopologicalSemiring α
    f : ι → α
    a : α
    inst✝ : T2Space α
    ⊢ Eq (tsum fun x => HDiv.hDiv (f x) a) (HDiv.hDiv (tsum fun x => f x) a)
  -/
  simpa only [div_eq_mul_inv] using tsum_mul_right
  /-
    🎉 no goals
  -/


theorem HasSum.const_div (h :  HasSum (fun x ↦ 1 / f x) a) (b : α) :
    HasSum (fun i ↦ b / f i) (b * a) := by
  /-
    ι : Type u_1
    α : Type u_3
    inst✝² : DivisionSemiring α
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSemiring α
    f : ι → α
    a : α
    h : HasSum (fun x => HDiv.hDiv 1 (f x)) a
    b : α
    ⊢ HasSum (fun i => HDiv.hDiv b (f i)) (HMul.hMul b a)
  -/
  have := h.mul_left b
  /-
    ι : Type u_1
    α : Type u_3
    inst✝² : DivisionSemiring α
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSemiring α
    f : ι → α
    a : α
    h : HasSum (fun x => HDiv.hDiv 1 (f x)) a
    b : α
    this : HasSum (fun i => HMul.hMul b (HDiv.hDiv 1 (f i))) (HMul.hMul b a)
    ⊢ HasSum (fun i => HDiv.hDiv b (f i)) (HMul.hMul b a)
  -/
  simpa only [div_eq_mul_inv, one_mul] using this
  /-
    🎉 no goals
  -/


theorem Summable.const_div (h : Summable (fun x ↦ 1 / f x)) (b : α) :
    Summable fun i ↦ b / f i :=
  (h.hasSum.const_div b).summable


theorem hasSum_const_div_iff (h : a₂ ≠ 0) :
    HasSum (fun i ↦ a₂ / f i) (a₂ * a₁) ↔ HasSum (1/ f) a₁ := by
  /-
    ι : Type u_1
    α : Type u_3
    inst✝² : DivisionSemiring α
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSemiring α
    f : ι → α
    a₁ a₂ : α
    h : Ne a₂ 0
    ⊢ Iff (HasSum (fun i => HDiv.hDiv a₂ (f i)) (HMul.hMul a₂ a₁)) (HasSum (HDiv.h …
  -/
  simpa only [div_eq_mul_inv, one_mul] using hasSum_mul_left_iff h
  /-
    🎉 no goals
  -/


theorem summable_const_div_iff (h : a ≠ 0) : (Summable fun i ↦ a / f i) ↔ Summable (1 / f) := by
  /-
    ι : Type u_1
    α : Type u_3
    inst✝² : DivisionSemiring α
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSemiring α
    f : ι → α
    a : α
    h : Ne a 0
    ⊢ Iff (Summable fun i => HDiv.hDiv a (f i)) (Summable (HDiv.hDiv 1 f))
  -/
  simpa only [div_eq_mul_inv, one_mul] using summable_mul_left_iff h
  /-
    🎉 no goals
  -/


theorem HasSum.mul_eq (hf : HasSum f s) (hg : HasSum g t)
    (hfg : HasSum (fun x : ι × κ ↦ f x.1 * g x.2) u) : s * t = u :=
  have key₁ : HasSum (fun i ↦ f i * t) (s * t) := hf.mul_right t
  have this : ∀ i : ι, HasSum (fun c : κ ↦ f i * g c) (f i * t) := fun i ↦ hg.mul_left (f i)
  have key₂ : HasSum (fun i ↦ f i * t) u := HasSum.prod_fiberwise hfg this
  key₁.unique key₂


theorem HasSum.mul (hf : HasSum f s) (hg : HasSum g t)
    (hfg : Summable fun x : ι × κ ↦ f x.1 * g x.2) :
    HasSum (fun x : ι × κ ↦ f x.1 * g x.2) (s * t) :=
  let ⟨_u, hu⟩ := hfg
  (hf.mul_eq hg hu).symm ▸ hu


/-- Product of two infinites sums indexed by arbitrary types.
    See also `tsum_mul_tsum_of_summable_norm` if `f` and `g` are absolutely summable. -/
theorem tsum_mul_tsum (hf : Summable f) (hg : Summable g)
    (hfg : Summable fun x : ι × κ ↦ f x.1 * g x.2) :
    ((∑' x, f x) * ∑' y, g y) = ∑' z : ι × κ, f z.1 * g z.2 :=
  hf.hasSum.mul_eq hg.hasSum hfg.hasSum


/-- The family `(k, l) : ℕ × ℕ ↦ f k * g l` is summable if and only if the family
`(n, k, l) : Σ (n : ℕ), antidiagonal n ↦ f k * g l` is summable. -/
theorem summable_mul_prod_iff_summable_mul_sigma_antidiagonal :
    (Summable fun x : A × A ↦ f x.1 * g x.2) ↔
      Summable fun x : Σn : A, antidiagonal n ↦ f (x.2 : A × A).1 * g (x.2 : A × A).2 :=
  Finset.sigmaAntidiagonalEquivProd.summable_iff.symm


theorem summable_sum_mul_antidiagonal_of_summable_mul
    (h : Summable fun x : A × A ↦ f x.1 * g x.2) :
    Summable fun n ↦ ∑ kl ∈ antidiagonal n, f kl.1 * g kl.2 := by
  /-
    α : Type u_3
    A : Type u_4
    inst✝⁵ : AddCommMonoid A
    inst✝⁴ : Finset.HasAntidiagonal A
    inst✝³ : TopologicalSpace α
    inst✝² : NonUnitalNonAssocSemiring α
    f g : A → α
    inst✝¹ : T3Space α
    inst✝ : TopologicalSemiring α
    h : Summable fun x => HMul.hMul (f x.1) (g x.2)
    ⊢ Summable fun n => (Finset.HasAntidiagonal.antidiagonal n).sum fun kl => HMul …
  -/
  rw [summable_mul_prod_iff_summable_mul_sigma_antidiagonal] at h
  /-
    α : Type u_3
    A : Type u_4
    inst✝⁵ : AddCommMonoid A
    inst✝⁴ : Finset.HasAntidiagonal A
    inst✝³ : TopologicalSpace α
    inst✝² : NonUnitalNonAssocSemiring α
    f g : A → α
    inst✝¹ : T3Space α
    inst✝ : TopologicalSemiring α
    h : Summable fun x => HMul.hMul (f (↑x.snd).1) (g (↑x.snd).2)
    ⊢ Summable fun n => (Finset.HasAntidiagonal.antidiagonal n).sum fun kl => HMul …
  -/
  conv => congr; ext; rw [← Finset.sum_finset_coe, ← tsum_fintype]
  /-
    α : Type u_3
    A : Type u_4
    inst✝⁵ : AddCommMonoid A
    inst✝⁴ : Finset.HasAntidiagonal A
    inst✝³ : TopologicalSpace α
    inst✝² : NonUnitalNonAssocSemiring α
    f g : A → α
    inst✝¹ : T3Space α
    inst✝ : TopologicalSemiring α
    h : Summable fun x => HMul.hMul (f (↑x.snd).1) (g (↑x.snd).2)
    ⊢ Summable fun x => tsum fun b => HMul.hMul (f (↑b).1) (g (↑b).2)
  -/
  exact h.sigma' fun n ↦ (hasSum_fintype _).summable
  /-
    🎉 no goals
  -/


/-- The **Cauchy product formula** for the product of two infinites sums indexed by `ℕ`, expressed
by summing on `Finset.antidiagonal`.

See also `tsum_mul_tsum_eq_tsum_sum_antidiagonal_of_summable_norm` if `f` and `g` are absolutely
summable. -/
theorem tsum_mul_tsum_eq_tsum_sum_antidiagonal (hf : Summable f) (hg : Summable g)
    (hfg : Summable fun x : A × A ↦ f x.1 * g x.2) :
    ((∑' n, f n) * ∑' n, g n) = ∑' n, ∑ kl ∈ antidiagonal n, f kl.1 * g kl.2 := by
  /-
    α : Type u_3
    A : Type u_4
    inst✝⁵ : AddCommMonoid A
    inst✝⁴ : Finset.HasAntidiagonal A
    inst✝³ : TopologicalSpace α
    inst✝² : NonUnitalNonAssocSemiring α
    f g : A → α
    inst✝¹ : T3Space α
    inst✝ : TopologicalSemiring α
    hf : Summable f
    hg : Summable g
    hfg : Summable fun x => HMul.hMul (f x.1) (g x.2)
    ⊢ Eq (HMul.hMul (tsum fun n => f n) (tsum fun n => g n)) (tsum fun n => (Finse …
  -/
  conv_rhs => congr; ext; rw [← Finset.sum_finset_coe, ← tsum_fintype]
  /-
    α : Type u_3
    A : Type u_4
    inst✝⁵ : AddCommMonoid A
    inst✝⁴ : Finset.HasAntidiagonal A
    inst✝³ : TopologicalSpace α
    inst✝² : NonUnitalNonAssocSemiring α
    f g : A → α
    inst✝¹ : T3Space α
    inst✝ : TopologicalSemiring α
    hf : Summable f
    hg : Summable g
    hfg : Summable fun x => HMul.hMul (f x.1) (g x.2)
    ⊢ Eq (HMul.hMul (tsum fun n => f n) (tsum fun n => g n)) (tsum fun x => tsum f …
  -/
  rw [tsum_mul_tsum hf hg hfg, ← sigmaAntidiagonalEquivProd.tsum_eq (_ : A × A → α)]
  exact
    tsum_sigma' (fun n ↦ (hasSum_fintype _).summable)
      (summable_mul_prod_iff_summable_mul_sigma_antidiagonal.mp hfg)


theorem summable_sum_mul_range_of_summable_mul (h : Summable fun x : ℕ × ℕ ↦ f x.1 * g x.2) :
    Summable fun n ↦ ∑ k ∈ range (n + 1), f k * g (n - k) := by
  /-
    α : Type u_3
    inst✝³ : TopologicalSpace α
    inst✝² : NonUnitalNonAssocSemiring α
    f g : Nat → α
    inst✝¹ : T3Space α
    inst✝ : TopologicalSemiring α
    h : Summable fun x => HMul.hMul (f x.1) (g x.2)
    ⊢ Summable fun n => (Finset.range (HAdd.hAdd n 1)).sum fun k => HMul.hMul (f k …
  -/
  simp_rw [← Nat.sum_antidiagonal_eq_sum_range_succ fun k l ↦ f k * g l]
  /-
    α : Type u_3
    inst✝³ : TopologicalSpace α
    inst✝² : NonUnitalNonAssocSemiring α
    f g : Nat → α
    inst✝¹ : T3Space α
    inst✝ : TopologicalSemiring α
    h : Summable fun x => HMul.hMul (f x.1) (g x.2)
    ⊢ Summable fun n => (Finset.HasAntidiagonal.antidiagonal n).sum fun ij => HMul …
  -/
  exact summable_sum_mul_antidiagonal_of_summable_mul h
  /-
    🎉 no goals
  -/


/-- The **Cauchy product formula** for the product of two infinites sums indexed by `ℕ`, expressed
by summing on `Finset.range`.

See also `tsum_mul_tsum_eq_tsum_sum_range_of_summable_norm` if `f` and `g` are absolutely summable.
-/
theorem tsum_mul_tsum_eq_tsum_sum_range (hf : Summable f) (hg : Summable g)
    (hfg : Summable fun x : ℕ × ℕ ↦ f x.1 * g x.2) :
    ((∑' n, f n) * ∑' n, g n) = ∑' n, ∑ k ∈ range (n + 1), f k * g (n - k) := by
  /-
    α : Type u_3
    inst✝³ : TopologicalSpace α
    inst✝² : NonUnitalNonAssocSemiring α
    f g : Nat → α
    inst✝¹ : T3Space α
    inst✝ : TopologicalSemiring α
    hf : Summable f
    hg : Summable g
    hfg : Summable fun x => HMul.hMul (f x.1) (g x.2)
    ⊢ Eq (HMul.hMul (tsum fun n => f n) (tsum fun n => g n)) (tsum fun n => (Finse …
  -/
  simp_rw [← Nat.sum_antidiagonal_eq_sum_range_succ fun k l ↦ f k * g l]
  /-
    α : Type u_3
    inst✝³ : TopologicalSpace α
    inst✝² : NonUnitalNonAssocSemiring α
    f g : Nat → α
    inst✝¹ : T3Space α
    inst✝ : TopologicalSemiring α
    hf : Summable f
    hg : Summable g
    hfg : Summable fun x => HMul.hMul (f x.1) (g x.2)
    ⊢ Eq (HMul.hMul (tsum fun n => f n) (tsum fun n => g n)) (tsum fun n => (Finse …
  -/
  exact tsum_mul_tsum_eq_tsum_sum_antidiagonal hf hg hfg
  /-
    🎉 no goals
  -/


