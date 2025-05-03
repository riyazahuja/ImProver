/-- `slope f a b = (b - a)⁻¹ • (f b -ᵥ f a)` is the slope of a function `f` on the interval
`[a, b]`. Note that `slope f a a = 0`, not the derivative of `f` at `a`. -/
def slope (f : k → PE) (a b : k) : E :=
  (b - a)⁻¹ • (f b -ᵥ f a)


theorem slope_fun_def (f : k → PE) : slope f = fun a b => (b - a)⁻¹ • (f b -ᵥ f a) :=
  rfl


theorem slope_def_field (f : k → k) (a b : k) : slope f a b = (f b - f a) / (b - a) :=
  (div_eq_inv_mul _ _).symm


theorem slope_fun_def_field (f : k → k) (a : k) : slope f a = fun b => (f b - f a) / (b - a) :=
  (div_eq_inv_mul _ _).symm


@[simp]
theorem slope_same (f : k → PE) (a : k) : (slope f a a : E) = 0 := by
  /-
    k : Type u_1
    E : Type u_2
    PE : Type u_3
    inst✝³ : Field k
    inst✝² : AddCommGroup E
    inst✝¹ : Module k E
    inst✝ : AddTorsor E PE
    f : k → PE
    a : k
    ⊢ Eq (slope f a a) 0
  -/
  rw [slope, sub_self, inv_zero, zero_smul]
  /-
    🎉 no goals
  -/


theorem slope_def_module (f : k → E) (a b : k) : slope f a b = (b - a)⁻¹ • (f b - f a) :=
  rfl


@[simp]
theorem sub_smul_slope (f : k → PE) (a b : k) : (b - a) • slope f a b = f b -ᵥ f a := by
  /-
    k : Type u_1
    E : Type u_2
    PE : Type u_3
    inst✝³ : Field k
    inst✝² : AddCommGroup E
    inst✝¹ : Module k E
    inst✝ : AddTorsor E PE
    f : k → PE
    a b : k
    ⊢ Eq (HSMul.hSMul (HSub.hSub b a) (slope f a b)) (VSub.vsub (f b) (f a))
  -/
  rcases eq_or_ne a b with (rfl | hne)
    /-
      case inl
      k : Type u_1
      E : Type u_2
      PE : Type u_3
      inst✝³ : Field k
      inst✝² : AddCommGroup E
      inst✝¹ : Module k E
      inst✝ : AddTorsor E PE
      f : k → PE
      a : k
      ⊢ Eq (HSMul.hSMul (HSub.hSub a a) (slope f a a)) (VSub.vsub (f a) (f a))
    -/
  · rw [sub_self, zero_smul, vsub_self]
    /-
      🎉 no goals
    -/
    /-
      case inr
      k : Type u_1
      E : Type u_2
      PE : Type u_3
      inst✝³ : Field k
      inst✝² : AddCommGroup E
      inst✝¹ : Module k E
      inst✝ : AddTorsor E PE
      f : k → PE
      a b : k
      hne : Ne a b
      ⊢ Eq (HSMul.hSMul (HSub.hSub b a) (slope f a b)) (VSub.vsub (f b) (f a))
    -/
  · rw [slope, smul_inv_smul₀ (sub_ne_zero.2 hne.symm)]
    /-
      🎉 no goals
    -/


theorem sub_smul_slope_vadd (f : k → PE) (a b : k) : (b - a) • slope f a b +ᵥ f a = f b := by
  /-
    k : Type u_1
    E : Type u_2
    PE : Type u_3
    inst✝³ : Field k
    inst✝² : AddCommGroup E
    inst✝¹ : Module k E
    inst✝ : AddTorsor E PE
    f : k → PE
    a b : k
    ⊢ Eq (HVAdd.hVAdd (HSMul.hSMul (HSub.hSub b a) (slope f a b)) (f a)) (f b)
  -/
  rw [sub_smul_slope, vsub_vadd]
  /-
    🎉 no goals
  -/


@[simp]
theorem slope_vadd_const (f : k → E) (c : PE) : (slope fun x => f x +ᵥ c) = slope f := by
  /-
    k : Type u_1
    E : Type u_2
    PE : Type u_3
    inst✝³ : Field k
    inst✝² : AddCommGroup E
    inst✝¹ : Module k E
    inst✝ : AddTorsor E PE
    f : k → E
    c : PE
    ⊢ Eq (slope fun x => HVAdd.hVAdd (f x) c) (slope f)
  -/
  ext a b
  /-
    case h.h
    k : Type u_1
    E : Type u_2
    PE : Type u_3
    inst✝³ : Field k
    inst✝² : AddCommGroup E
    inst✝¹ : Module k E
    inst✝ : AddTorsor E PE
    f : k → E
    c : PE
    a b : k
    ⊢ Eq (slope (fun x => HVAdd.hVAdd (f x) c) a b) (slope f a b)
  -/
  simp only [slope, vadd_vsub_vadd_cancel_right, vsub_eq_sub]
  /-
    🎉 no goals
  -/


@[simp]
theorem slope_sub_smul (f : k → E) {a b : k} (h : a ≠ b) :
    slope (fun x => (x - a) • f x) a b = f b := by
  /-
    k : Type u_1
    E : Type u_2
    inst✝² : Field k
    inst✝¹ : AddCommGroup E
    inst✝ : Module k E
    f : k → E
    a b : k
    h : Ne a b
    ⊢ Eq (slope (fun x => HSMul.hSMul (HSub.hSub x a) (f x)) a b) (f b)
  -/
  simp [slope, inv_smul_smul₀ (sub_ne_zero.2 h.symm)]
  /-
    🎉 no goals
  -/


theorem eq_of_slope_eq_zero {f : k → PE} {a b : k} (h : slope f a b = (0 : E)) : f a = f b := by
  /-
    k : Type u_1
    E : Type u_2
    PE : Type u_3
    inst✝³ : Field k
    inst✝² : AddCommGroup E
    inst✝¹ : Module k E
    inst✝ : AddTorsor E PE
    f : k → PE
    a b : k
    h : Eq (slope f a b) 0
    ⊢ Eq (f a) (f b)
  -/
  rw [← sub_smul_slope_vadd f a b, h, smul_zero, zero_vadd]
  /-
    🎉 no goals
  -/


theorem AffineMap.slope_comp {F PF : Type*} [AddCommGroup F] [Module k F] [AddTorsor F PF]
    (f : PE →ᵃ[k] PF) (g : k → PE) (a b : k) : slope (f ∘ g) a b = f.linear (slope g a b) := by
  /-
    k : Type u_1
    E : Type u_2
    PE : Type u_3
    inst✝⁶ : Field k
    inst✝⁵ : AddCommGroup E
    inst✝⁴ : Module k E
    inst✝³ : AddTorsor E PE
    F : Type u_4
    PF : Type u_5
    inst✝² : AddCommGroup F
    inst✝¹ : Module k F
    inst✝ : AddTorsor F PF
    f : AffineMap k PE PF
    g : k → PE
    a b : k
    ⊢ Eq (slope (Function.comp (⇑f) g) a b) (f.linear (slope g a b))
  -/
  simp only [slope, (· ∘ ·), f.linear.map_smul, f.linearMap_vsub]
  /-
    🎉 no goals
  -/


theorem LinearMap.slope_comp {F : Type*} [AddCommGroup F] [Module k F] (f : E →ₗ[k] F) (g : k → E)
    (a b : k) : slope (f ∘ g) a b = f (slope g a b) :=
  f.toAffineMap.slope_comp g a b


theorem slope_comm (f : k → PE) (a b : k) : slope f a b = slope f b a := by
  /-
    k : Type u_1
    E : Type u_2
    PE : Type u_3
    inst✝³ : Field k
    inst✝² : AddCommGroup E
    inst✝¹ : Module k E
    inst✝ : AddTorsor E PE
    f : k → PE
    a b : k
    ⊢ Eq (slope f a b) (slope f b a)
  -/
  rw [slope, slope, ← neg_vsub_eq_vsub_rev, smul_neg, ← neg_smul, neg_inv, neg_sub]
  /-
    🎉 no goals
  -/


@[simp] lemma slope_neg (f : k → E) (x y : k) : slope (fun t ↦ -f t) x y = -slope f x y := by
  /-
    k : Type u_1
    E : Type u_2
    inst✝² : Field k
    inst✝¹ : AddCommGroup E
    inst✝ : Module k E
    f : k → E
    x y : k
    ⊢ Eq (slope (fun t => Neg.neg (f t)) x y) (Neg.neg (slope f x y))
  -/
  simp only [slope_def_module, neg_sub_neg, ← smul_neg, neg_sub]
  /-
    🎉 no goals
  -/


@[simp] lemma slope_neg_fun (f : k → E) : slope (-f) = -slope f := by
  /-
    k : Type u_1
    E : Type u_2
    inst✝² : Field k
    inst✝¹ : AddCommGroup E
    inst✝ : Module k E
    f : k → E
    ⊢ Eq (slope (Neg.neg f)) (Neg.neg (slope f))
  -/
  ext x y; exact slope_neg f x y
           /-
             🎉 no goals
           -/


/-- `slope f a c` is a linear combination of `slope f a b` and `slope f b c`. This version
explicitly provides coefficients. If `a ≠ c`, then the sum of the coefficients is `1`, so it is
actually an affine combination, see `lineMap_slope_slope_sub_div_sub`. -/
theorem sub_div_sub_smul_slope_add_sub_div_sub_smul_slope (f : k → PE) (a b c : k) :
    ((b - a) / (c - a)) • slope f a b + ((c - b) / (c - a)) • slope f b c = slope f a c := by
  /-
    k : Type u_1
    E : Type u_2
    PE : Type u_3
    inst✝³ : Field k
    inst✝² : AddCommGroup E
    inst✝¹ : Module k E
    inst✝ : AddTorsor E PE
    f : k → PE
    a b c : k
    ⊢ Eq (HAdd.hAdd (HSMul.hSMul (HDiv.hDiv (HSub.hSub b a) (HSub.hSub c a)) (slop …
  -/
  by_cases hab : a = b
    /-
      case pos
      k : Type u_1
      E : Type u_2
      PE : Type u_3
      inst✝³ : Field k
      inst✝² : AddCommGroup E
      inst✝¹ : Module k E
      inst✝ : AddTorsor E PE
      f : k → PE
      a b c : k
      hab : Eq a b
      ⊢ Eq (HAdd.hAdd (HSMul.hSMul (HDiv.hDiv (HSub.hSub b a) (HSub.hSub c a)) (slop …
    -/
  · subst hab
    /-
      case pos
      k : Type u_1
      E : Type u_2
      PE : Type u_3
      inst✝³ : Field k
      inst✝² : AddCommGroup E
      inst✝¹ : Module k E
      inst✝ : AddTorsor E PE
      f : k → PE
      a c : k
      ⊢ Eq (HAdd.hAdd (HSMul.hSMul (HDiv.hDiv (HSub.hSub a a) (HSub.hSub c a)) (slop …
    -/
    rw [sub_self, zero_div, zero_smul, zero_add]
    /-
      case pos
      k : Type u_1
      E : Type u_2
      PE : Type u_3
      inst✝³ : Field k
      inst✝² : AddCommGroup E
      inst✝¹ : Module k E
      inst✝ : AddTorsor E PE
      f : k → PE
      a c : k
      ⊢ Eq (HSMul.hSMul (HDiv.hDiv (HSub.hSub c a) (HSub.hSub c a)) (slope f a c)) ( …
    -/
    by_cases hac : a = c
      /-
        case pos
        k : Type u_1
        E : Type u_2
        PE : Type u_3
        inst✝³ : Field k
        inst✝² : AddCommGroup E
        inst✝¹ : Module k E
        inst✝ : AddTorsor E PE
        f : k → PE
        a c : k
        hac : Eq a c
        ⊢ Eq (HSMul.hSMul (HDiv.hDiv (HSub.hSub c a) (HSub.hSub c a)) (slope f a c)) ( …
      -/
    · simp [hac]
      /-
        🎉 no goals
      -/
      /-
        case neg
        k : Type u_1
        E : Type u_2
        PE : Type u_3
        inst✝³ : Field k
        inst✝² : AddCommGroup E
        inst✝¹ : Module k E
        inst✝ : AddTorsor E PE
        f : k → PE
        a c : k
        hac : Not (Eq a c)
        ⊢ Eq (HSMul.hSMul (HDiv.hDiv (HSub.hSub c a) (HSub.hSub c a)) (slope f a c)) ( …
      -/
    · rw [div_self (sub_ne_zero.2 <| Ne.symm hac), one_smul]
      /-
        🎉 no goals
      -/
  /-
    case neg
    k : Type u_1
    E : Type u_2
    PE : Type u_3
    inst✝³ : Field k
    inst✝² : AddCommGroup E
    inst✝¹ : Module k E
    inst✝ : AddTorsor E PE
    f : k → PE
    a b c : k
    hab : Not (Eq a b)
    ⊢ Eq (HAdd.hAdd (HSMul.hSMul (HDiv.hDiv (HSub.hSub b a) (HSub.hSub c a)) (slop …
  -/
  by_cases hbc : b = c
    /-
      case pos
      k : Type u_1
      E : Type u_2
      PE : Type u_3
      inst✝³ : Field k
      inst✝² : AddCommGroup E
      inst✝¹ : Module k E
      inst✝ : AddTorsor E PE
      f : k → PE
      a b c : k
      hab : Not (Eq a b)
      hbc : Eq b c
      ⊢ Eq (HAdd.hAdd (HSMul.hSMul (HDiv.hDiv (HSub.hSub b a) (HSub.hSub c a)) (slop …
    -/
  · subst hbc
    /-
      case pos
      k : Type u_1
      E : Type u_2
      PE : Type u_3
      inst✝³ : Field k
      inst✝² : AddCommGroup E
      inst✝¹ : Module k E
      inst✝ : AddTorsor E PE
      f : k → PE
      a b : k
      hab : Not (Eq a b)
      ⊢ Eq (HAdd.hAdd (HSMul.hSMul (HDiv.hDiv (HSub.hSub b a) (HSub.hSub b a)) (slop …
    -/
    simp [sub_ne_zero.2 (Ne.symm hab)]
    /-
      🎉 no goals
    -/
  /-
    case neg
    k : Type u_1
    E : Type u_2
    PE : Type u_3
    inst✝³ : Field k
    inst✝² : AddCommGroup E
    inst✝¹ : Module k E
    inst✝ : AddTorsor E PE
    f : k → PE
    a b c : k
    hab : Not (Eq a b)
    hbc : Not (Eq b c)
    ⊢ Eq (HAdd.hAdd (HSMul.hSMul (HDiv.hDiv (HSub.hSub b a) (HSub.hSub c a)) (slop …
  -/
  rw [add_comm]
  simp_rw [slope, div_eq_inv_mul, mul_smul, ← smul_add,
    smul_inv_smul₀ (sub_ne_zero.2 <| Ne.symm hab), smul_inv_smul₀ (sub_ne_zero.2 <| Ne.symm hbc),
    vsub_add_vsub_cancel]


/-- `slope f a c` is an affine combination of `slope f a b` and `slope f b c`. This version uses
`lineMap` to express this property. -/
theorem lineMap_slope_slope_sub_div_sub (f : k → PE) (a b c : k) (h : a ≠ c) :
    lineMap (slope f a b) (slope f b c) ((c - b) / (c - a)) = slope f a c := by
  field_simp [sub_ne_zero.2 h.symm, ← sub_div_sub_smul_slope_add_sub_div_sub_smul_slope f a b c,
    lineMap_apply_module]


/-- `slope f a b` is an affine combination of `slope f a (lineMap a b r)` and
`slope f (lineMap a b r) b`. We use `lineMap` to express this property. -/
theorem lineMap_slope_lineMap_slope_lineMap (f : k → PE) (a b r : k) :
    lineMap (slope f (lineMap a b r) b) (slope f a (lineMap a b r)) r = slope f a b := by
  /-
    k : Type u_1
    E : Type u_2
    PE : Type u_3
    inst✝³ : Field k
    inst✝² : AddCommGroup E
    inst✝¹ : Module k E
    inst✝ : AddTorsor E PE
    f : k → PE
    a b r : k
    ⊢ Eq ((AffineMap.lineMap (slope f ((AffineMap.lineMap a b) r) b) (slope f a (( …
  -/
  obtain rfl | hab : a = b ∨ a ≠ b := Classical.em _; · simp
                                                        /-
                                                          🎉 no goals
                                                        -/
  /-
    case inr
    k : Type u_1
    E : Type u_2
    PE : Type u_3
    inst✝³ : Field k
    inst✝² : AddCommGroup E
    inst✝¹ : Module k E
    inst✝ : AddTorsor E PE
    f : k → PE
    a b r : k
    hab : Ne a b
    ⊢ Eq ((AffineMap.lineMap (slope f ((AffineMap.lineMap a b) r) b) (slope f a (( …
  -/
  rw [slope_comm _ a, slope_comm _ a, slope_comm _ _ b]
  /-
    case inr
    k : Type u_1
    E : Type u_2
    PE : Type u_3
    inst✝³ : Field k
    inst✝² : AddCommGroup E
    inst✝¹ : Module k E
    inst✝ : AddTorsor E PE
    f : k → PE
    a b r : k
    hab : Ne a b
    ⊢ Eq ((AffineMap.lineMap (slope f b ((AffineMap.lineMap a b) r)) (slope f ((Af …
  -/
  convert lineMap_slope_slope_sub_div_sub f b (lineMap a b r) a hab.symm using 2
  rw [lineMap_apply_ring, eq_div_iff (sub_ne_zero.2 hab), sub_mul, one_mul, mul_sub, ← sub_sub,
    sub_sub_cancel]

