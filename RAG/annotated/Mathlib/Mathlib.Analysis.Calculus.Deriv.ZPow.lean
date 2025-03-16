theorem hasStrictDerivAt_zpow (m : ℤ) (x : 𝕜) (h : x ≠ 0 ∨ 0 ≤ m) :
    HasStrictDerivAt (fun x => x ^ m) ((m : 𝕜) * x ^ (m - 1)) x := by
  have : ∀ m : ℤ, 0 < m → HasStrictDerivAt (· ^ m) ((m : 𝕜) * x ^ (m - 1)) x := fun m hm ↦ by
    lift m to ℕ using hm.le
    simp only [zpow_natCast, Int.cast_natCast]
    convert hasStrictDerivAt_pow m x using 2
    rw [← Int.ofNat_one, ← Int.ofNat_sub, zpow_natCast]
    norm_cast at hm
  /-
    𝕜 : Type u
    inst✝ : NontriviallyNormedField 𝕜
    m : Int
    x : 𝕜
    h : Or (Ne x 0) (LE.le 0 m)
    this : ∀ (m : Int), LT.lt 0 m → HasStrictDerivAt (fun x => HPow.hPow x m) (HMu …
    ⊢ HasStrictDerivAt (fun x => HPow.hPow x m) (HMul.hMul (↑m) (HPow.hPow x (HSub …
  -/
  rcases lt_trichotomy m 0 with (hm | hm | hm)
    /-
      case inl
      𝕜 : Type u
      inst✝ : NontriviallyNormedField 𝕜
      m : Int
      x : 𝕜
      h : Or (Ne x 0) (LE.le 0 m)
      this : ∀ (m : Int), LT.lt 0 m → HasStrictDerivAt (fun x => HPow.hPow x m) (HMu …
      hm : LT.lt m 0
      ⊢ HasStrictDerivAt (fun x => HPow.hPow x m) (HMul.hMul (↑m) (HPow.hPow x (HSub …
    -/
  · have hx : x ≠ 0 := h.resolve_right hm.not_le
    have := (hasStrictDerivAt_inv ?_).scomp _ (this (-m) (neg_pos.2 hm)) <;>
      [skip; exact zpow_ne_zero _ hx]
    /-
      case inl.refine_2
      𝕜 : Type u
      inst✝ : NontriviallyNormedField 𝕜
      m : Int
      x : 𝕜
      h : Or (Ne x 0) (LE.le 0 m)
      this✝ : ∀ (m : Int), LT.lt 0 m → HasStrictDerivAt (fun x => HPow.hPow x m) (HM …
      hm : LT.lt m 0
      hx : Ne x 0
      this : HasStrictDerivAt (Function.comp Inv.inv fun x => HPow.hPow x (Neg.neg m …
      ⊢ HasStrictDerivAt (fun x => HPow.hPow x m) (HMul.hMul (↑m) (HPow.hPow x (HSub …
    -/
    simp only [Function.comp_def, zpow_neg, one_div, inv_inv, smul_eq_mul] at this
    /-
      case inl.refine_2
      𝕜 : Type u
      inst✝ : NontriviallyNormedField 𝕜
      m : Int
      x : 𝕜
      h : Or (Ne x 0) (LE.le 0 m)
      this✝ : ∀ (m : Int), LT.lt 0 m → HasStrictDerivAt (fun x => HPow.hPow x m) (HM …
      hm : LT.lt m 0
      hx : Ne x 0
      this : HasStrictDerivAt (fun x => HPow.hPow x m) (HMul.hMul (HMul.hMul (↑(Neg. …
      ⊢ HasStrictDerivAt (fun x => HPow.hPow x m) (HMul.hMul (↑m) (HPow.hPow x (HSub …
    -/
    convert this using 1
    rw [sq, mul_inv, inv_inv, Int.cast_neg, neg_mul, neg_mul_neg, ← zpow_add₀ hx, mul_assoc, ←
      zpow_add₀ hx]
    /-
      case h.e'_9
      𝕜 : Type u
      inst✝ : NontriviallyNormedField 𝕜
      m : Int
      x : 𝕜
      h : Or (Ne x 0) (LE.le 0 m)
      this✝ : ∀ (m : Int), LT.lt 0 m → HasStrictDerivAt (fun x => HPow.hPow x m) (HM …
      hm : LT.lt m 0
      hx : Ne x 0
      this : HasStrictDerivAt (fun x => HPow.hPow x m) (HMul.hMul (HMul.hMul (↑(Neg. …
      ⊢ Eq (HMul.hMul (↑m) (HPow.hPow x (HSub.hSub m 1))) (HMul.hMul (↑m) (HPow.hPow …
    -/
    congr
    /-
      case h.e'_9.e_a.e_a
      𝕜 : Type u
      inst✝ : NontriviallyNormedField 𝕜
      m : Int
      x : 𝕜
      h : Or (Ne x 0) (LE.le 0 m)
      this✝ : ∀ (m : Int), LT.lt 0 m → HasStrictDerivAt (fun x => HPow.hPow x m) (HM …
      hm : LT.lt m 0
      hx : Ne x 0
      this : HasStrictDerivAt (fun x => HPow.hPow x m) (HMul.hMul (HMul.hMul (↑(Neg. …
      ⊢ Eq (HSub.hSub m 1) (HAdd.hAdd (HSub.hSub (Neg.neg m) 1) (HAdd.hAdd m m))
    -/
    /-
      🎉 no goals
    -/
    abel
    /-
      🎉 no goals
    -/
    /-
      case inr.inl
      𝕜 : Type u
      inst✝ : NontriviallyNormedField 𝕜
      m : Int
      x : 𝕜
      h : Or (Ne x 0) (LE.le 0 m)
      this : ∀ (m : Int), LT.lt 0 m → HasStrictDerivAt (fun x => HPow.hPow x m) (HMu …
      hm : Eq m 0
      ⊢ HasStrictDerivAt (fun x => HPow.hPow x m) (HMul.hMul (↑m) (HPow.hPow x (HSub …
    -/
  · simp only [hm, zpow_zero, Int.cast_zero, zero_mul, hasStrictDerivAt_const]
    /-
      🎉 no goals
    -/
    /-
      case inr.inr
      𝕜 : Type u
      inst✝ : NontriviallyNormedField 𝕜
      m : Int
      x : 𝕜
      h : Or (Ne x 0) (LE.le 0 m)
      this : ∀ (m : Int), LT.lt 0 m → HasStrictDerivAt (fun x => HPow.hPow x m) (HMu …
      hm : LT.lt 0 m
      ⊢ HasStrictDerivAt (fun x => HPow.hPow x m) (HMul.hMul (↑m) (HPow.hPow x (HSub …
    -/
  · exact this m hm
    /-
      🎉 no goals
    -/


theorem hasDerivAt_zpow (m : ℤ) (x : 𝕜) (h : x ≠ 0 ∨ 0 ≤ m) :
    HasDerivAt (fun x => x ^ m) ((m : 𝕜) * x ^ (m - 1)) x :=
  (hasStrictDerivAt_zpow m x h).hasDerivAt


theorem hasDerivWithinAt_zpow (m : ℤ) (x : 𝕜) (h : x ≠ 0 ∨ 0 ≤ m) (s : Set 𝕜) :
    HasDerivWithinAt (fun x => x ^ m) ((m : 𝕜) * x ^ (m - 1)) s x :=
  (hasDerivAt_zpow m x h).hasDerivWithinAt


theorem differentiableAt_zpow : DifferentiableAt 𝕜 (fun x => x ^ m) x ↔ x ≠ 0 ∨ 0 ≤ m :=
  ⟨fun H => NormedField.continuousAt_zpow.1 H.continuousAt, fun H =>
    (hasDerivAt_zpow m x H).differentiableAt⟩


theorem differentiableWithinAt_zpow (m : ℤ) (x : 𝕜) (h : x ≠ 0 ∨ 0 ≤ m) :
    DifferentiableWithinAt 𝕜 (fun x => x ^ m) s x :=
  (differentiableAt_zpow.mpr h).differentiableWithinAt


theorem differentiableOn_zpow (m : ℤ) (s : Set 𝕜) (h : (0 : 𝕜) ∉ s ∨ 0 ≤ m) :
    DifferentiableOn 𝕜 (fun x => x ^ m) s := fun x hxs =>
  differentiableWithinAt_zpow m x <| h.imp_left <| ne_of_mem_of_not_mem hxs


theorem deriv_zpow (m : ℤ) (x : 𝕜) : deriv (fun x => x ^ m) x = m * x ^ (m - 1) := by
  /-
    𝕜 : Type u
    inst✝ : NontriviallyNormedField 𝕜
    m : Int
    x : 𝕜
    ⊢ Eq (deriv (fun x => HPow.hPow x m) x) (HMul.hMul (↑m) (HPow.hPow x (HSub.hSu …
  -/
  by_cases H : x ≠ 0 ∨ 0 ≤ m
    /-
      case pos
      𝕜 : Type u
      inst✝ : NontriviallyNormedField 𝕜
      m : Int
      x : 𝕜
      H : Or (Ne x 0) (LE.le 0 m)
      ⊢ Eq (deriv (fun x => HPow.hPow x m) x) (HMul.hMul (↑m) (HPow.hPow x (HSub.hSu …
    -/
  · exact (hasDerivAt_zpow m x H).deriv
    /-
      🎉 no goals
    -/
    /-
      case neg
      𝕜 : Type u
      inst✝ : NontriviallyNormedField 𝕜
      m : Int
      x : 𝕜
      H : Not (Or (Ne x 0) (LE.le 0 m))
      ⊢ Eq (deriv (fun x => HPow.hPow x m) x) (HMul.hMul (↑m) (HPow.hPow x (HSub.hSu …
    -/
  · rw [deriv_zero_of_not_differentiableAt (mt differentiableAt_zpow.1 H)]
    /-
      case neg
      𝕜 : Type u
      inst✝ : NontriviallyNormedField 𝕜
      m : Int
      x : 𝕜
      H : Not (Or (Ne x 0) (LE.le 0 m))
      ⊢ Eq 0 (HMul.hMul (↑m) (HPow.hPow x (HSub.hSub m 1)))
    -/
    push_neg at H
    /-
      case neg
      𝕜 : Type u
      inst✝ : NontriviallyNormedField 𝕜
      m : Int
      x : 𝕜
      H : And (Eq x 0) (LT.lt m 0)
      ⊢ Eq 0 (HMul.hMul (↑m) (HPow.hPow x (HSub.hSub m 1)))
    -/
    rcases H with ⟨rfl, hm⟩
    /-
      case neg.intro
      𝕜 : Type u
      inst✝ : NontriviallyNormedField 𝕜
      m : Int
      hm : LT.lt m 0
      ⊢ Eq 0 (HMul.hMul (↑m) (HPow.hPow 0 (HSub.hSub m 1)))
    -/
    rw [zero_zpow _ ((sub_one_lt _).trans hm).ne, mul_zero]
    /-
      🎉 no goals
    -/


@[simp]
theorem deriv_zpow' (m : ℤ) : (deriv fun x : 𝕜 => x ^ m) = fun x => (m : 𝕜) * x ^ (m - 1) :=
  funext <| deriv_zpow m


theorem derivWithin_zpow (hxs : UniqueDiffWithinAt 𝕜 s x) (h : x ≠ 0 ∨ 0 ≤ m) :
    derivWithin (fun x => x ^ m) s x = (m : 𝕜) * x ^ (m - 1) :=
  (hasDerivWithinAt_zpow m x h s).derivWithin hxs


@[simp]
theorem iter_deriv_zpow' (m : ℤ) (k : ℕ) :
    (deriv^[k] fun x : 𝕜 => x ^ m) =
      fun x => (∏ i ∈ Finset.range k, ((m : 𝕜) - i)) * x ^ (m - k) := by
  /-
    𝕜 : Type u
    inst✝ : NontriviallyNormedField 𝕜
    m : Int
    k : Nat
    ⊢ Eq (Nat.iterate deriv k fun x => HPow.hPow x m) fun x => HMul.hMul ((Finset. …
  -/
  induction' k with k ihk
  · simp only [one_mul, Int.ofNat_zero, id, sub_zero, Finset.prod_range_zero,
      Function.iterate_zero]
  · simp only [Function.iterate_succ_apply', ihk, deriv_const_mul_field', deriv_zpow',
      Finset.prod_range_succ, Int.ofNat_succ, ← sub_sub, Int.cast_sub, Int.cast_natCast, mul_assoc]


theorem iter_deriv_zpow (m : ℤ) (x : 𝕜) (k : ℕ) :
    deriv^[k] (fun y => y ^ m) x = (∏ i ∈ Finset.range k, ((m : 𝕜) - i)) * x ^ (m - k) :=
  congr_fun (iter_deriv_zpow' m k) x


theorem iter_deriv_pow (n : ℕ) (x : 𝕜) (k : ℕ) :
    deriv^[k] (fun x : 𝕜 => x ^ n) x = (∏ i ∈ Finset.range k, ((n : 𝕜) - i)) * x ^ (n - k) := by
  /-
    𝕜 : Type u
    inst✝ : NontriviallyNormedField 𝕜
    n : Nat
    x : 𝕜
    k : Nat
    ⊢ Eq (Nat.iterate deriv k (fun x => HPow.hPow x n) x) (HMul.hMul ((Finset.rang …
  -/
  simp only [← zpow_natCast, iter_deriv_zpow, Int.cast_natCast]
  /-
    𝕜 : Type u
    inst✝ : NontriviallyNormedField 𝕜
    n : Nat
    x : 𝕜
    k : Nat
    ⊢ Eq (HMul.hMul ((Finset.range k).prod fun x => HSub.hSub ↑n ↑x) (HPow.hPow x  …
  -/
  rcases le_or_lt k n with hkn | hnk
    /-
      case inl
      𝕜 : Type u
      inst✝ : NontriviallyNormedField 𝕜
      n : Nat
      x : 𝕜
      k : Nat
      hkn : LE.le k n
      ⊢ Eq (HMul.hMul ((Finset.range k).prod fun x => HSub.hSub ↑n ↑x) (HPow.hPow x  …
    -/
  · rw [Int.ofNat_sub hkn]
    /-
      🎉 no goals
    -/
  · have : (∏ i ∈ Finset.range k, (n - i : 𝕜)) = 0 :=
      Finset.prod_eq_zero (Finset.mem_range.2 hnk) (sub_self _)
    /-
      case inr
      𝕜 : Type u
      inst✝ : NontriviallyNormedField 𝕜
      n : Nat
      x : 𝕜
      k : Nat
      hnk : LT.lt n k
      this : Eq ((Finset.range k).prod fun i => HSub.hSub ↑n ↑i) 0
      ⊢ Eq (HMul.hMul ((Finset.range k).prod fun x => HSub.hSub ↑n ↑x) (HPow.hPow x  …
    -/
    simp only [this, zero_mul]
    /-
      🎉 no goals
    -/


@[simp]
theorem iter_deriv_pow' (n k : ℕ) :
    (deriv^[k] fun x : 𝕜 => x ^ n) =
      fun x => (∏ i ∈ Finset.range k, ((n : 𝕜) - i)) * x ^ (n - k) :=
  funext fun x => iter_deriv_pow n x k


theorem iter_deriv_inv (k : ℕ) (x : 𝕜) :
    deriv^[k] Inv.inv x = (∏ i ∈ Finset.range k, (-1 - i : 𝕜)) * x ^ (-1 - k : ℤ) := by
  /-
    𝕜 : Type u
    inst✝ : NontriviallyNormedField 𝕜
    k : Nat
    x : 𝕜
    ⊢ Eq (Nat.iterate deriv k Inv.inv x) (HMul.hMul ((Finset.range k).prod fun i = …
  -/
  simpa only [zpow_neg_one, Int.cast_neg, Int.cast_one] using iter_deriv_zpow (-1) x k
  /-
    🎉 no goals
  -/


@[simp]
theorem iter_deriv_inv' (k : ℕ) :
    deriv^[k] Inv.inv = fun x : 𝕜 => (∏ i ∈ Finset.range k, (-1 - i : 𝕜)) * x ^ (-1 - k : ℤ) :=
  funext (iter_deriv_inv k)


theorem DifferentiableWithinAt.zpow (hf : DifferentiableWithinAt 𝕜 f t a) (h : f a ≠ 0 ∨ 0 ≤ m) :
    DifferentiableWithinAt 𝕜 (fun x => f x ^ m) t a :=
  (differentiableAt_zpow.2 h).comp_differentiableWithinAt a hf


theorem DifferentiableAt.zpow (hf : DifferentiableAt 𝕜 f a) (h : f a ≠ 0 ∨ 0 ≤ m) :
    DifferentiableAt 𝕜 (fun x => f x ^ m) a :=
  (differentiableAt_zpow.2 h).comp a hf


theorem DifferentiableOn.zpow (hf : DifferentiableOn 𝕜 f t) (h : (∀ x ∈ t, f x ≠ 0) ∨ 0 ≤ m) :
    DifferentiableOn 𝕜 (fun x => f x ^ m) t := fun x hx =>
  (hf x hx).zpow <| h.imp_left fun h => h x hx


theorem Differentiable.zpow (hf : Differentiable 𝕜 f) (h : (∀ x, f x ≠ 0) ∨ 0 ≤ m) :
    Differentiable 𝕜 fun x => f x ^ m := fun x => (hf x).zpow <| h.imp_left fun h => h x

