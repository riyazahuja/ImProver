/-- Divide by `of' k G g`, discarding terms not divisible by this. -/
noncomputable def divOf (x : k[G]) (g : G) : k[G] :=
  -- note: comapping by `+ g` has the effect of subtracting `g` from every element in
  -- the support, and discarding the elements of the support from which `g` can't be subtracted.
  -- If `G` is an additive group, such as `ℤ` when used for `LaurentPolynomial`,
  -- then no discarding occurs.
  @Finsupp.comapDomain.addMonoidHom _ _ _ _ (g + ·) (add_right_injective g) x


local infixl:70 " /ᵒᶠ " => divOf


@[simp]
theorem divOf_apply (g : G) (x : k[G]) (g' : G) : (x /ᵒᶠ g) g' = x (g + g') :=
  rfl


@[simp]
theorem support_divOf (g : G) (x : k[G]) :
    (x /ᵒᶠ g).support =
      x.support.preimage (g + ·) (Function.Injective.injOn (add_right_injective g)) :=
  rfl


@[simp]
theorem zero_divOf (g : G) : (0 : k[G]) /ᵒᶠ g = 0 :=
  map_zero (Finsupp.comapDomain.addMonoidHom _)


@[simp]
theorem divOf_zero (x : k[G]) : x /ᵒᶠ 0 = x := by
  /-
    k : Type u_1
    G : Type u_2
    inst✝¹ : Semiring k
    inst✝ : AddCancelCommMonoid G
    x : AddMonoidAlgebra k G
    ⊢ Eq (x.divOf 0) x
  -/
  refine Finsupp.ext fun _ => ?_  -- Porting note (https://github.com/leanprover-community/mathlib4/issues/11041): `ext` doesn't work
  /-
    k : Type u_1
    G : Type u_2
    inst✝¹ : Semiring k
    inst✝ : AddCancelCommMonoid G
    x : AddMonoidAlgebra k G
    x✝ : G
    ⊢ Eq ((x.divOf 0) x✝) (x x✝)
  -/
  simp only [AddMonoidAlgebra.divOf_apply, zero_add]
  /-
    🎉 no goals
  -/


theorem add_divOf (x y : k[G]) (g : G) : (x + y) /ᵒᶠ g = x /ᵒᶠ g + y /ᵒᶠ g :=
  map_add (Finsupp.comapDomain.addMonoidHom _) _ _


theorem divOf_add (x : k[G]) (a b : G) : x /ᵒᶠ (a + b) = x /ᵒᶠ a /ᵒᶠ b := by
  /-
    k : Type u_1
    G : Type u_2
    inst✝¹ : Semiring k
    inst✝ : AddCancelCommMonoid G
    x : AddMonoidAlgebra k G
    a b : G
    ⊢ Eq (x.divOf (HAdd.hAdd a b)) ((x.divOf a).divOf b)
  -/
  refine Finsupp.ext fun _ => ?_  -- Porting note (https://github.com/leanprover-community/mathlib4/issues/11041): `ext` doesn't work
  /-
    k : Type u_1
    G : Type u_2
    inst✝¹ : Semiring k
    inst✝ : AddCancelCommMonoid G
    x : AddMonoidAlgebra k G
    a b x✝ : G
    ⊢ Eq ((x.divOf (HAdd.hAdd a b)) x✝) (((x.divOf a).divOf b) x✝)
  -/
  simp only [AddMonoidAlgebra.divOf_apply, add_assoc]
  /-
    🎉 no goals
  -/


/-- A bundled version of `AddMonoidAlgebra.divOf`. -/
@[simps]
noncomputable def divOfHom : Multiplicative G →* AddMonoid.End k[G] where
  toFun g :=
    { toFun := fun x => divOf x g.toAdd
      map_zero' := zero_divOf _
      map_add' := fun x y => add_divOf x y g.toAdd }
  map_one' := AddMonoidHom.ext divOf_zero
  map_mul' g₁ g₂ :=
    AddMonoidHom.ext fun _x =>
      (congr_arg _ (add_comm g₁.toAdd g₂.toAdd)).trans
        (divOf_add _ _ _)


theorem of'_mul_divOf (a : G) (x : k[G]) : of' k G a * x /ᵒᶠ a = x := by
  /-
    k : Type u_1
    G : Type u_2
    inst✝¹ : Semiring k
    inst✝ : AddCancelCommMonoid G
    a : G
    x : AddMonoidAlgebra k G
    ⊢ Eq ((HMul.hMul (AddMonoidAlgebra.of' k G a) x).divOf a) x
  -/
  refine Finsupp.ext fun _ => ?_  -- Porting note (https://github.com/leanprover-community/mathlib4/issues/11041): `ext` doesn't work
  /-
    k : Type u_1
    G : Type u_2
    inst✝¹ : Semiring k
    inst✝ : AddCancelCommMonoid G
    a : G
    x : AddMonoidAlgebra k G
    x✝ : G
    ⊢ Eq (((HMul.hMul (AddMonoidAlgebra.of' k G a) x).divOf a) x✝) (x x✝)
  -/
  rw [AddMonoidAlgebra.divOf_apply, of'_apply, single_mul_apply_aux, one_mul]
  /-
    case H
    k : Type u_1
    G : Type u_2
    inst✝¹ : Semiring k
    inst✝ : AddCancelCommMonoid G
    a : G
    x : AddMonoidAlgebra k G
    x✝ : G
    ⊢ ∀ (a_1 : G), Iff (Eq (HAdd.hAdd a a_1) (HAdd.hAdd a x✝)) (Eq a_1 x✝)
  -/
  intro c
  /-
    case H
    k : Type u_1
    G : Type u_2
    inst✝¹ : Semiring k
    inst✝ : AddCancelCommMonoid G
    a : G
    x : AddMonoidAlgebra k G
    x✝ c : G
    ⊢ Iff (Eq (HAdd.hAdd a c) (HAdd.hAdd a x✝)) (Eq c x✝)
  -/
  exact add_right_inj _
  /-
    🎉 no goals
  -/


theorem mul_of'_divOf (x : k[G]) (a : G) : x * of' k G a /ᵒᶠ a = x := by
  /-
    k : Type u_1
    G : Type u_2
    inst✝¹ : Semiring k
    inst✝ : AddCancelCommMonoid G
    x : AddMonoidAlgebra k G
    a : G
    ⊢ Eq ((HMul.hMul x (AddMonoidAlgebra.of' k G a)).divOf a) x
  -/
  refine Finsupp.ext fun _ => ?_  -- Porting note (https://github.com/leanprover-community/mathlib4/issues/11041): `ext` doesn't work
  /-
    k : Type u_1
    G : Type u_2
    inst✝¹ : Semiring k
    inst✝ : AddCancelCommMonoid G
    x : AddMonoidAlgebra k G
    a x✝ : G
    ⊢ Eq (((HMul.hMul x (AddMonoidAlgebra.of' k G a)).divOf a) x✝) (x x✝)
  -/
  rw [AddMonoidAlgebra.divOf_apply, of'_apply, mul_single_apply_aux, mul_one]
  /-
    case H
    k : Type u_1
    G : Type u_2
    inst✝¹ : Semiring k
    inst✝ : AddCancelCommMonoid G
    x : AddMonoidAlgebra k G
    a x✝ : G
    ⊢ ∀ (a_1 : G), Iff (Eq (HAdd.hAdd a_1 a) (HAdd.hAdd a x✝)) (Eq a_1 x✝)
  -/
  intro c
  /-
    case H
    k : Type u_1
    G : Type u_2
    inst✝¹ : Semiring k
    inst✝ : AddCancelCommMonoid G
    x : AddMonoidAlgebra k G
    a x✝ c : G
    ⊢ Iff (Eq (HAdd.hAdd c a) (HAdd.hAdd a x✝)) (Eq c x✝)
  -/
  rw [add_comm]
  /-
    case H
    k : Type u_1
    G : Type u_2
    inst✝¹ : Semiring k
    inst✝ : AddCancelCommMonoid G
    x : AddMonoidAlgebra k G
    a x✝ c : G
    ⊢ Iff (Eq (HAdd.hAdd a c) (HAdd.hAdd a x✝)) (Eq c x✝)
  -/
  exact add_right_inj _
  /-
    🎉 no goals
  -/


theorem of'_divOf (a : G) : of' k G a /ᵒᶠ a = 1 := by
  /-
    k : Type u_1
    G : Type u_2
    inst✝¹ : Semiring k
    inst✝ : AddCancelCommMonoid G
    a : G
    ⊢ Eq ((AddMonoidAlgebra.of' k G a).divOf a) 1
  -/
  simpa only [one_mul] using mul_of'_divOf (1 : k[G]) a
  /-
    🎉 no goals
  -/


/-- The remainder upon division by `of' k G g`. -/
noncomputable def modOf (x : k[G]) (g : G) : k[G] :=
  letI := Classical.decPred fun g₁ => ∃ g₂, g₁ = g + g₂
  x.filter fun g₁ => ¬∃ g₂, g₁ = g + g₂


local infixl:70 " %ᵒᶠ " => modOf


@[simp]
theorem modOf_apply_of_not_exists_add (x : k[G]) (g : G) (g' : G)
    (h : ¬∃ d, g' = g + d) : (x %ᵒᶠ g) g' = x g' := by
  /-
    k : Type u_1
    G : Type u_2
    inst✝¹ : Semiring k
    inst✝ : AddCancelCommMonoid G
    x : AddMonoidAlgebra k G
    g g' : G
    h : Not (Exists fun d => Eq g' (HAdd.hAdd g d))
    ⊢ Eq ((x.modOf g) g') (x g')
  -/
  classical exact Finsupp.filter_apply_pos _ _ h
  /-
    🎉 no goals
  -/


@[simp]
theorem modOf_apply_of_exists_add (x : k[G]) (g : G) (g' : G)
    (h : ∃ d, g' = g + d) : (x %ᵒᶠ g) g' = 0 := by
  /-
    k : Type u_1
    G : Type u_2
    inst✝¹ : Semiring k
    inst✝ : AddCancelCommMonoid G
    x : AddMonoidAlgebra k G
    g g' : G
    h : Exists fun d => Eq g' (HAdd.hAdd g d)
    ⊢ Eq ((x.modOf g) g') 0
  -/
  classical exact Finsupp.filter_apply_neg _ _ <| by rwa [Classical.not_not]
  /-
    🎉 no goals
  -/


@[simp]
theorem modOf_apply_add_self (x : k[G]) (g : G) (d : G) : (x %ᵒᶠ g) (d + g) = 0 :=
  modOf_apply_of_exists_add _ _ _ ⟨_, add_comm _ _⟩


theorem modOf_apply_self_add (x : k[G]) (g : G) (d : G) : (x %ᵒᶠ g) (g + d) = 0 :=
  modOf_apply_of_exists_add _ _ _ ⟨_, rfl⟩


theorem of'_mul_modOf (g : G) (x : k[G]) : of' k G g * x %ᵒᶠ g = 0 := by
  /-
    k : Type u_1
    G : Type u_2
    inst✝¹ : Semiring k
    inst✝ : AddCancelCommMonoid G
    g : G
    x : AddMonoidAlgebra k G
    ⊢ Eq ((HMul.hMul (AddMonoidAlgebra.of' k G g) x).modOf g) 0
  -/
  refine Finsupp.ext fun g' => ?_  -- Porting note (https://github.com/leanprover-community/mathlib4/issues/11041): `ext g'` doesn't work
  /-
    k : Type u_1
    G : Type u_2
    inst✝¹ : Semiring k
    inst✝ : AddCancelCommMonoid G
    g : G
    x : AddMonoidAlgebra k G
    g' : G
    ⊢ Eq (((HMul.hMul (AddMonoidAlgebra.of' k G g) x).modOf g) g') (0 g')
  -/
  rw [Finsupp.zero_apply]
  /-
    k : Type u_1
    G : Type u_2
    inst✝¹ : Semiring k
    inst✝ : AddCancelCommMonoid G
    g : G
    x : AddMonoidAlgebra k G
    g' : G
    ⊢ Eq (((HMul.hMul (AddMonoidAlgebra.of' k G g) x).modOf g) g') 0
  -/
  obtain ⟨d, rfl⟩ | h := em (∃ d, g' = g + d)
    /-
      case inl.intro
      k : Type u_1
      G : Type u_2
      inst✝¹ : Semiring k
      inst✝ : AddCancelCommMonoid G
      g : G
      x : AddMonoidAlgebra k G
      d : G
      ⊢ Eq (((HMul.hMul (AddMonoidAlgebra.of' k G g) x).modOf g) (HAdd.hAdd g d)) 0
    -/
  · rw [modOf_apply_self_add]
    /-
      🎉 no goals
    -/
    /-
      case inr
      k : Type u_1
      G : Type u_2
      inst✝¹ : Semiring k
      inst✝ : AddCancelCommMonoid G
      g : G
      x : AddMonoidAlgebra k G
      g' : G
      h : Not (Exists fun d => Eq g' (HAdd.hAdd g d))
      ⊢ Eq (((HMul.hMul (AddMonoidAlgebra.of' k G g) x).modOf g) g') 0
    -/
  · rw [modOf_apply_of_not_exists_add _ _ _ h, of'_apply, single_mul_apply_of_not_exists_add _ _ h]
    /-
      🎉 no goals
    -/


theorem mul_of'_modOf (x : k[G]) (g : G) : x * of' k G g %ᵒᶠ g = 0 := by
  /-
    k : Type u_1
    G : Type u_2
    inst✝¹ : Semiring k
    inst✝ : AddCancelCommMonoid G
    x : AddMonoidAlgebra k G
    g : G
    ⊢ Eq ((HMul.hMul x (AddMonoidAlgebra.of' k G g)).modOf g) 0
  -/
  refine Finsupp.ext fun g' => ?_  -- Porting note (https://github.com/leanprover-community/mathlib4/issues/11041): `ext g'` doesn't work
  /-
    k : Type u_1
    G : Type u_2
    inst✝¹ : Semiring k
    inst✝ : AddCancelCommMonoid G
    x : AddMonoidAlgebra k G
    g g' : G
    ⊢ Eq (((HMul.hMul x (AddMonoidAlgebra.of' k G g)).modOf g) g') (0 g')
  -/
  rw [Finsupp.zero_apply]
  /-
    k : Type u_1
    G : Type u_2
    inst✝¹ : Semiring k
    inst✝ : AddCancelCommMonoid G
    x : AddMonoidAlgebra k G
    g g' : G
    ⊢ Eq (((HMul.hMul x (AddMonoidAlgebra.of' k G g)).modOf g) g') 0
  -/
  obtain ⟨d, rfl⟩ | h := em (∃ d, g' = g + d)
    /-
      case inl.intro
      k : Type u_1
      G : Type u_2
      inst✝¹ : Semiring k
      inst✝ : AddCancelCommMonoid G
      x : AddMonoidAlgebra k G
      g d : G
      ⊢ Eq (((HMul.hMul x (AddMonoidAlgebra.of' k G g)).modOf g) (HAdd.hAdd g d)) 0
    -/
  · rw [modOf_apply_self_add]
    /-
      🎉 no goals
    -/
    /-
      case inr
      k : Type u_1
      G : Type u_2
      inst✝¹ : Semiring k
      inst✝ : AddCancelCommMonoid G
      x : AddMonoidAlgebra k G
      g g' : G
      h : Not (Exists fun d => Eq g' (HAdd.hAdd g d))
      ⊢ Eq (((HMul.hMul x (AddMonoidAlgebra.of' k G g)).modOf g) g') 0
    -/
  · rw [modOf_apply_of_not_exists_add _ _ _ h, of'_apply, mul_single_apply_of_not_exists_add]
    /-
      case inr.h
      k : Type u_1
      G : Type u_2
      inst✝¹ : Semiring k
      inst✝ : AddCancelCommMonoid G
      x : AddMonoidAlgebra k G
      g g' : G
      h : Not (Exists fun d => Eq g' (HAdd.hAdd g d))
      ⊢ Not (Exists fun d => Eq g' (HAdd.hAdd d g))
    -/
    simpa only [add_comm] using h
    /-
      🎉 no goals
    -/


theorem of'_modOf (g : G) : of' k G g %ᵒᶠ g = 0 := by
  /-
    k : Type u_1
    G : Type u_2
    inst✝¹ : Semiring k
    inst✝ : AddCancelCommMonoid G
    g : G
    ⊢ Eq ((AddMonoidAlgebra.of' k G g).modOf g) 0
  -/
  simpa only [one_mul] using mul_of'_modOf (1 : k[G]) g
  /-
    🎉 no goals
  -/


theorem divOf_add_modOf (x : k[G]) (g : G) :
    of' k G g * (x /ᵒᶠ g) + x %ᵒᶠ g = x := by
  /-
    k : Type u_1
    G : Type u_2
    inst✝¹ : Semiring k
    inst✝ : AddCancelCommMonoid G
    x : AddMonoidAlgebra k G
    g : G
    ⊢ Eq (HAdd.hAdd (HMul.hMul (AddMonoidAlgebra.of' k G g) (x.divOf g)) (x.modOf  …
  -/
  refine Finsupp.ext fun g' => ?_  -- Porting note (https://github.com/leanprover-community/mathlib4/issues/11041): `ext` doesn't work
  /-
    k : Type u_1
    G : Type u_2
    inst✝¹ : Semiring k
    inst✝ : AddCancelCommMonoid G
    x : AddMonoidAlgebra k G
    g g' : G
    ⊢ Eq ((HAdd.hAdd (HMul.hMul (AddMonoidAlgebra.of' k G g) (x.divOf g)) (x.modOf …
  -/
  rw [Finsupp.add_apply] -- Porting note: changed from `simp_rw` which can't see through the type
  /-
    k : Type u_1
    G : Type u_2
    inst✝¹ : Semiring k
    inst✝ : AddCancelCommMonoid G
    x : AddMonoidAlgebra k G
    g g' : G
    ⊢ Eq (HAdd.hAdd ((HMul.hMul (AddMonoidAlgebra.of' k G g) (x.divOf g)) g') ((x. …
  -/
  obtain ⟨d, rfl⟩ | h := em (∃ d, g' = g + d)
  /-
    case inl.intro
    k : Type u_1
    G : Type u_2
    inst✝¹ : Semiring k
    inst✝ : AddCancelCommMonoid G
    x : AddMonoidAlgebra k G
    g d : G
    ⊢ Eq (HAdd.hAdd ((HMul.hMul (AddMonoidAlgebra.of' k G g) (x.divOf g)) (HAdd.hA …
  -/
  swap
  · rw [modOf_apply_of_not_exists_add x _ _ h, of'_apply, single_mul_apply_of_not_exists_add _ _ h,
      zero_add]
    /-
      case inl.intro
      k : Type u_1
      G : Type u_2
      inst✝¹ : Semiring k
      inst✝ : AddCancelCommMonoid G
      x : AddMonoidAlgebra k G
      g d : G
      ⊢ Eq (HAdd.hAdd ((HMul.hMul (AddMonoidAlgebra.of' k G g) (x.divOf g)) (HAdd.hA …
    -/
  · rw [modOf_apply_self_add, add_zero]
    /-
      case inl.intro
      k : Type u_1
      G : Type u_2
      inst✝¹ : Semiring k
      inst✝ : AddCancelCommMonoid G
      x : AddMonoidAlgebra k G
      g d : G
      ⊢ Eq ((HMul.hMul (AddMonoidAlgebra.of' k G g) (x.divOf g)) (HAdd.hAdd g d)) (x …
    -/
    rw [of'_apply, single_mul_apply_aux _ _ _, one_mul, divOf_apply]
    /-
      case inl.intro.H
      k : Type u_1
      G : Type u_2
      inst✝¹ : Semiring k
      inst✝ : AddCancelCommMonoid G
      x : AddMonoidAlgebra k G
      g d : G
      ⊢ ∀ (a : G), Iff (Eq (HAdd.hAdd g a) (HAdd.hAdd g d)) (Eq a d)
    -/
    intro a
    /-
      case inl.intro.H
      k : Type u_1
      G : Type u_2
      inst✝¹ : Semiring k
      inst✝ : AddCancelCommMonoid G
      x : AddMonoidAlgebra k G
      g d a : G
      ⊢ Iff (Eq (HAdd.hAdd g a) (HAdd.hAdd g d)) (Eq a d)
    -/
    exact add_right_inj _
    /-
      🎉 no goals
    -/


theorem modOf_add_divOf (x : k[G]) (g : G) : x %ᵒᶠ g + of' k G g * (x /ᵒᶠ g) = x := by
  /-
    k : Type u_1
    G : Type u_2
    inst✝¹ : Semiring k
    inst✝ : AddCancelCommMonoid G
    x : AddMonoidAlgebra k G
    g : G
    ⊢ Eq (HAdd.hAdd (x.modOf g) (HMul.hMul (AddMonoidAlgebra.of' k G g) (x.divOf g …
  -/
  rw [add_comm, divOf_add_modOf]
  /-
    🎉 no goals
  -/


theorem of'_dvd_iff_modOf_eq_zero {x : k[G]} {g : G} :
    of' k G g ∣ x ↔ x %ᵒᶠ g = 0 := by
  /-
    k : Type u_1
    G : Type u_2
    inst✝¹ : Semiring k
    inst✝ : AddCancelCommMonoid G
    x : AddMonoidAlgebra k G
    g : G
    ⊢ Iff (Dvd.dvd (AddMonoidAlgebra.of' k G g) x) (Eq (x.modOf g) 0)
  -/
  constructor
    /-
      case mp
      k : Type u_1
      G : Type u_2
      inst✝¹ : Semiring k
      inst✝ : AddCancelCommMonoid G
      x : AddMonoidAlgebra k G
      g : G
      ⊢ Dvd.dvd (AddMonoidAlgebra.of' k G g) x → Eq (x.modOf g) 0
    -/
  · rintro ⟨x, rfl⟩
    /-
      case mp.intro
      k : Type u_1
      G : Type u_2
      inst✝¹ : Semiring k
      inst✝ : AddCancelCommMonoid G
      g : G
      x : AddMonoidAlgebra k G
      ⊢ Eq ((HMul.hMul (AddMonoidAlgebra.of' k G g) x).modOf g) 0
    -/
    rw [of'_mul_modOf]
    /-
      🎉 no goals
    -/
    /-
      case mpr
      k : Type u_1
      G : Type u_2
      inst✝¹ : Semiring k
      inst✝ : AddCancelCommMonoid G
      x : AddMonoidAlgebra k G
      g : G
      ⊢ Eq (x.modOf g) 0 → Dvd.dvd (AddMonoidAlgebra.of' k G g) x
    -/
  · intro h
    /-
      case mpr
      k : Type u_1
      G : Type u_2
      inst✝¹ : Semiring k
      inst✝ : AddCancelCommMonoid G
      x : AddMonoidAlgebra k G
      g : G
      h : Eq (x.modOf g) 0
      ⊢ Dvd.dvd (AddMonoidAlgebra.of' k G g) x
    -/
    rw [← divOf_add_modOf x g, h, add_zero]
    /-
      case mpr
      k : Type u_1
      G : Type u_2
      inst✝¹ : Semiring k
      inst✝ : AddCancelCommMonoid G
      x : AddMonoidAlgebra k G
      g : G
      h : Eq (x.modOf g) 0
      ⊢ Dvd.dvd (AddMonoidAlgebra.of' k G g) (HMul.hMul (AddMonoidAlgebra.of' k G g) …
    -/
    exact dvd_mul_right _ _
    /-
      🎉 no goals
    -/


