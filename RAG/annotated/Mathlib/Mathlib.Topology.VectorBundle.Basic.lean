/-- A mixin class for `Pretrivialization`, stating that a pretrivialization is fiberwise linear with
respect to given module structures on its fibers and the model fiber. -/
protected class Pretrivialization.IsLinear [AddCommMonoid F] [Module R F] [∀ x, AddCommMonoid (E x)]
  [∀ x, Module R (E x)] (e : Pretrivialization F (π F E)) : Prop where
  linear : ∀ b ∈ e.baseSet, IsLinearMap R fun x : E b => (e ⟨b, x⟩).2


theorem linear [AddCommMonoid F] [Module R F] [∀ x, AddCommMonoid (E x)] [∀ x, Module R (E x)]
    [e.IsLinear R] {b : B} (hb : b ∈ e.baseSet) :
    IsLinearMap R fun x : E b => (e ⟨b, x⟩).2 :=
  Pretrivialization.IsLinear.linear b hb


/-- A fiberwise linear inverse to `e`. -/
@[simps!]
protected def symmₗ (e : Pretrivialization F (π F E)) [e.IsLinear R] (b : B) : F →ₗ[R] E b := by
  /-
    R : Type u_1
    B : Type u_2
    F : Type u_3
    E : B → Type u_4
    inst✝⁷ : Semiring R
    inst✝⁶ : TopologicalSpace F
    inst✝⁵ : TopologicalSpace B
    e✝ : Pretrivialization F Bundle.TotalSpace.proj
    x : Bundle.TotalSpace F E
    b✝ : B
    y : E b✝
    inst✝⁴ : AddCommMonoid F
    inst✝³ : Module R F
    inst✝² : (x : B) → AddCommMonoid (E x)
    inst✝¹ : (x : B) → Module R (E x)
    e : Pretrivialization F Bundle.TotalSpace.proj
    inst✝ : Pretrivialization.IsLinear R e
    b : B
    ⊢ LinearMap (RingHom.id R) F (E b)
  -/
  refine IsLinearMap.mk' (e.symm b) ?_
  /-
    R : Type u_1
    B : Type u_2
    F : Type u_3
    E : B → Type u_4
    inst✝⁷ : Semiring R
    inst✝⁶ : TopologicalSpace F
    inst✝⁵ : TopologicalSpace B
    e✝ : Pretrivialization F Bundle.TotalSpace.proj
    x : Bundle.TotalSpace F E
    b✝ : B
    y : E b✝
    inst✝⁴ : AddCommMonoid F
    inst✝³ : Module R F
    inst✝² : (x : B) → AddCommMonoid (E x)
    inst✝¹ : (x : B) → Module R (E x)
    e : Pretrivialization F Bundle.TotalSpace.proj
    inst✝ : Pretrivialization.IsLinear R e
    b : B
    ⊢ IsLinearMap R (e.symm b)
  -/
  by_cases hb : b ∈ e.baseSet
  · exact (((e.linear R hb).mk' _).inverse (e.symm b) (e.symm_apply_apply_mk hb) fun v ↦
      congr_arg Prod.snd <| e.apply_mk_symm hb v).isLinear
    /-
      case neg
      R : Type u_1
      B : Type u_2
      F : Type u_3
      E : B → Type u_4
      inst✝⁷ : Semiring R
      inst✝⁶ : TopologicalSpace F
      inst✝⁵ : TopologicalSpace B
      e✝ : Pretrivialization F Bundle.TotalSpace.proj
      x : Bundle.TotalSpace F E
      b✝ : B
      y : E b✝
      inst✝⁴ : AddCommMonoid F
      inst✝³ : Module R F
      inst✝² : (x : B) → AddCommMonoid (E x)
      inst✝¹ : (x : B) → Module R (E x)
      e : Pretrivialization F Bundle.TotalSpace.proj
      inst✝ : Pretrivialization.IsLinear R e
      b : B
      hb : Not (Membership.mem e.baseSet b)
      ⊢ IsLinearMap R (e.symm b)
    -/
  · rw [e.coe_symm_of_not_mem hb]
    /-
      case neg
      R : Type u_1
      B : Type u_2
      F : Type u_3
      E : B → Type u_4
      inst✝⁷ : Semiring R
      inst✝⁶ : TopologicalSpace F
      inst✝⁵ : TopologicalSpace B
      e✝ : Pretrivialization F Bundle.TotalSpace.proj
      x : Bundle.TotalSpace F E
      b✝ : B
      y : E b✝
      inst✝⁴ : AddCommMonoid F
      inst✝³ : Module R F
      inst✝² : (x : B) → AddCommMonoid (E x)
      inst✝¹ : (x : B) → Module R (E x)
      e : Pretrivialization F Bundle.TotalSpace.proj
      inst✝ : Pretrivialization.IsLinear R e
      b : B
      hb : Not (Membership.mem e.baseSet b)
      ⊢ IsLinearMap R 0
    -/
    exact (0 : F →ₗ[R] E b).isLinear
    /-
      🎉 no goals
    -/


/-- A pretrivialization for a vector bundle defines linear equivalences between the
fibers and the model space. -/
@[simps (config := .asFn)]
def linearEquivAt (e : Pretrivialization F (π F E)) [e.IsLinear R] (b : B) (hb : b ∈ e.baseSet) :
    E b ≃ₗ[R] F where
  toFun y := (e ⟨b, y⟩).2
  invFun := e.symm b
  left_inv := e.symm_apply_apply_mk hb
                    /-
                      R : Type u_1
                      B : Type u_2
                      F : Type u_3
                      E : B → Type u_4
                      inst✝⁷ : Semiring R
                      inst✝⁶ : TopologicalSpace F
                      inst✝⁵ : TopologicalSpace B
                      e✝ : Pretrivialization F Bundle.TotalSpace.proj
                      x : Bundle.TotalSpace F E
                      b✝ : B
                      y : E b✝
                      inst✝⁴ : AddCommMonoid F
                      inst✝³ : Module R F
                      inst✝² : (x : B) → AddCommMonoid (E x)
                      inst✝¹ : (x : B) → Module R (E x)
                      e : Pretrivialization F Bundle.TotalSpace.proj
                      inst✝ : Pretrivialization.IsLinear R e
                      b : B
                      hb : Membership.mem e.baseSet b
                      v : F
                      ⊢ Eq ({ toFun := fun y => (↑e { proj := b, snd := y }).2, map_add' := ⋯, map_s …
                    -/
  right_inv v := by simp_rw [e.apply_mk_symm hb v]
                    /-
                      🎉 no goals
                    -/
  map_add' v w := (e.linear R hb).map_add v w
  map_smul' c v := (e.linear R hb).map_smul c v


open Classical in
/-- A fiberwise linear map equal to `e` on `e.baseSet`. -/
protected def linearMapAt (e : Pretrivialization F (π F E)) [e.IsLinear R] (b : B) : E b →ₗ[R] F :=
  if hb : b ∈ e.baseSet then e.linearEquivAt R b hb else 0


open Classical in
theorem coe_linearMapAt (e : Pretrivialization F (π F E)) [e.IsLinear R] (b : B) :
    ⇑(e.linearMapAt R b) = fun y => if b ∈ e.baseSet then (e ⟨b, y⟩).2 else 0 := by
  /-
    R : Type u_1
    B : Type u_2
    F : Type u_3
    E : B → Type u_4
    inst✝⁷ : Semiring R
    inst✝⁶ : TopologicalSpace F
    inst✝⁵ : TopologicalSpace B
    inst✝⁴ : AddCommMonoid F
    inst✝³ : Module R F
    inst✝² : (x : B) → AddCommMonoid (E x)
    inst✝¹ : (x : B) → Module R (E x)
    e : Pretrivialization F Bundle.TotalSpace.proj
    inst✝ : Pretrivialization.IsLinear R e
    b : B
    ⊢ Eq ⇑(Pretrivialization.linearMapAt R e b) fun y => ite (Membership.mem e.bas …
  -/
  rw [Pretrivialization.linearMapAt]
  /-
    R : Type u_1
    B : Type u_2
    F : Type u_3
    E : B → Type u_4
    inst✝⁷ : Semiring R
    inst✝⁶ : TopologicalSpace F
    inst✝⁵ : TopologicalSpace B
    inst✝⁴ : AddCommMonoid F
    inst✝³ : Module R F
    inst✝² : (x : B) → AddCommMonoid (E x)
    inst✝¹ : (x : B) → Module R (E x)
    e : Pretrivialization F Bundle.TotalSpace.proj
    inst✝ : Pretrivialization.IsLinear R e
    b : B
    ⊢ Eq ⇑(dite (Membership.mem e.baseSet b) (fun hb => ↑(Pretrivialization.linear …
  -/
                /-
                  🎉 no goals
                -/
  split_ifs <;> rfl
                /-
                  🎉 no goals
                -/


theorem coe_linearMapAt_of_mem (e : Pretrivialization F (π F E)) [e.IsLinear R] {b : B}
    (hb : b ∈ e.baseSet) : ⇑(e.linearMapAt R b) = fun y => (e ⟨b, y⟩).2 := by
  /-
    R : Type u_1
    B : Type u_2
    F : Type u_3
    E : B → Type u_4
    inst✝⁷ : Semiring R
    inst✝⁶ : TopologicalSpace F
    inst✝⁵ : TopologicalSpace B
    inst✝⁴ : AddCommMonoid F
    inst✝³ : Module R F
    inst✝² : (x : B) → AddCommMonoid (E x)
    inst✝¹ : (x : B) → Module R (E x)
    e : Pretrivialization F Bundle.TotalSpace.proj
    inst✝ : Pretrivialization.IsLinear R e
    b : B
    hb : Membership.mem e.baseSet b
    ⊢ Eq ⇑(Pretrivialization.linearMapAt R e b) fun y => (↑e { proj := b, snd := y …
  -/
  simp_rw [coe_linearMapAt, if_pos hb]
  /-
    🎉 no goals
  -/


open Classical in
theorem linearMapAt_apply (e : Pretrivialization F (π F E)) [e.IsLinear R] {b : B} (y : E b) :
    e.linearMapAt R b y = if b ∈ e.baseSet then (e ⟨b, y⟩).2 else 0 := by
  /-
    R : Type u_1
    B : Type u_2
    F : Type u_3
    E : B → Type u_4
    inst✝⁷ : Semiring R
    inst✝⁶ : TopologicalSpace F
    inst✝⁵ : TopologicalSpace B
    inst✝⁴ : AddCommMonoid F
    inst✝³ : Module R F
    inst✝² : (x : B) → AddCommMonoid (E x)
    inst✝¹ : (x : B) → Module R (E x)
    e : Pretrivialization F Bundle.TotalSpace.proj
    inst✝ : Pretrivialization.IsLinear R e
    b : B
    y : E b
    ⊢ Eq ((Pretrivialization.linearMapAt R e b) y) (ite (Membership.mem e.baseSet  …
  -/
  rw [coe_linearMapAt]
  /-
    🎉 no goals
  -/


theorem linearMapAt_def_of_mem (e : Pretrivialization F (π F E)) [e.IsLinear R] {b : B}
    (hb : b ∈ e.baseSet) : e.linearMapAt R b = e.linearEquivAt R b hb :=
  dif_pos hb


theorem linearMapAt_def_of_not_mem (e : Pretrivialization F (π F E)) [e.IsLinear R] {b : B}
    (hb : b ∉ e.baseSet) : e.linearMapAt R b = 0 :=
  dif_neg hb


theorem linearMapAt_eq_zero (e : Pretrivialization F (π F E)) [e.IsLinear R] {b : B}
    (hb : b ∉ e.baseSet) : e.linearMapAt R b = 0 :=
  dif_neg hb


theorem symmₗ_linearMapAt (e : Pretrivialization F (π F E)) [e.IsLinear R] {b : B}
    (hb : b ∈ e.baseSet) (y : E b) : e.symmₗ R b (e.linearMapAt R b y) = y := by
  /-
    R : Type u_1
    B : Type u_2
    F : Type u_3
    E : B → Type u_4
    inst✝⁷ : Semiring R
    inst✝⁶ : TopologicalSpace F
    inst✝⁵ : TopologicalSpace B
    inst✝⁴ : AddCommMonoid F
    inst✝³ : Module R F
    inst✝² : (x : B) → AddCommMonoid (E x)
    inst✝¹ : (x : B) → Module R (E x)
    e : Pretrivialization F Bundle.TotalSpace.proj
    inst✝ : Pretrivialization.IsLinear R e
    b : B
    hb : Membership.mem e.baseSet b
    y : E b
    ⊢ Eq ((Pretrivialization.symmₗ R e b) ((Pretrivialization.linearMapAt R e b) y …
  -/
  rw [e.linearMapAt_def_of_mem hb]
  /-
    R : Type u_1
    B : Type u_2
    F : Type u_3
    E : B → Type u_4
    inst✝⁷ : Semiring R
    inst✝⁶ : TopologicalSpace F
    inst✝⁵ : TopologicalSpace B
    inst✝⁴ : AddCommMonoid F
    inst✝³ : Module R F
    inst✝² : (x : B) → AddCommMonoid (E x)
    inst✝¹ : (x : B) → Module R (E x)
    e : Pretrivialization F Bundle.TotalSpace.proj
    inst✝ : Pretrivialization.IsLinear R e
    b : B
    hb : Membership.mem e.baseSet b
    y : E b
    ⊢ Eq ((Pretrivialization.symmₗ R e b) (↑(Pretrivialization.linearEquivAt R e b …
  -/
  exact (e.linearEquivAt R b hb).left_inv y
  /-
    🎉 no goals
  -/


theorem linearMapAt_symmₗ (e : Pretrivialization F (π F E)) [e.IsLinear R] {b : B}
    (hb : b ∈ e.baseSet) (y : F) : e.linearMapAt R b (e.symmₗ R b y) = y := by
  /-
    R : Type u_1
    B : Type u_2
    F : Type u_3
    E : B → Type u_4
    inst✝⁷ : Semiring R
    inst✝⁶ : TopologicalSpace F
    inst✝⁵ : TopologicalSpace B
    inst✝⁴ : AddCommMonoid F
    inst✝³ : Module R F
    inst✝² : (x : B) → AddCommMonoid (E x)
    inst✝¹ : (x : B) → Module R (E x)
    e : Pretrivialization F Bundle.TotalSpace.proj
    inst✝ : Pretrivialization.IsLinear R e
    b : B
    hb : Membership.mem e.baseSet b
    y : F
    ⊢ Eq ((Pretrivialization.linearMapAt R e b) ((Pretrivialization.symmₗ R e b) y …
  -/
  rw [e.linearMapAt_def_of_mem hb]
  /-
    R : Type u_1
    B : Type u_2
    F : Type u_3
    E : B → Type u_4
    inst✝⁷ : Semiring R
    inst✝⁶ : TopologicalSpace F
    inst✝⁵ : TopologicalSpace B
    inst✝⁴ : AddCommMonoid F
    inst✝³ : Module R F
    inst✝² : (x : B) → AddCommMonoid (E x)
    inst✝¹ : (x : B) → Module R (E x)
    e : Pretrivialization F Bundle.TotalSpace.proj
    inst✝ : Pretrivialization.IsLinear R e
    b : B
    hb : Membership.mem e.baseSet b
    y : F
    ⊢ Eq (↑(Pretrivialization.linearEquivAt R e b hb) ((Pretrivialization.symmₗ R  …
  -/
  exact (e.linearEquivAt R b hb).right_inv y
  /-
    🎉 no goals
  -/


/-- A mixin class for `Trivialization`, stating that a trivialization is fiberwise linear with
respect to given module structures on its fibers and the model fiber. -/
protected class Trivialization.IsLinear [AddCommMonoid F] [Module R F] [∀ x, AddCommMonoid (E x)]
  [∀ x, Module R (E x)] (e : Trivialization F (π F E)) : Prop where
  linear : ∀ b ∈ e.baseSet, IsLinearMap R fun x : E b => (e ⟨b, x⟩).2


protected theorem linear [AddCommMonoid F] [Module R F] [∀ x, AddCommMonoid (E x)]
    [∀ x, Module R (E x)] [e.IsLinear R] {b : B} (hb : b ∈ e.baseSet) :
    IsLinearMap R fun y : E b => (e ⟨b, y⟩).2 :=
  Trivialization.IsLinear.linear b hb


instance toPretrivialization.isLinear [AddCommMonoid F] [Module R F] [∀ x, AddCommMonoid (E x)]
    [∀ x, Module R (E x)] [e.IsLinear R] : e.toPretrivialization.IsLinear R :=
  { (‹_› : e.IsLinear R) with }


/-- A trivialization for a vector bundle defines linear equivalences between the
fibers and the model space. -/
def linearEquivAt (e : Trivialization F (π F E)) [e.IsLinear R] (b : B) (hb : b ∈ e.baseSet) :
    E b ≃ₗ[R] F :=
  e.toPretrivialization.linearEquivAt R b hb


@[simp]
theorem linearEquivAt_apply (e : Trivialization F (π F E)) [e.IsLinear R] (b : B)
    (hb : b ∈ e.baseSet) (v : E b) : e.linearEquivAt R b hb v = (e ⟨b, v⟩).2 :=
  rfl


@[simp]
theorem linearEquivAt_symm_apply (e : Trivialization F (π F E)) [e.IsLinear R] (b : B)
    (hb : b ∈ e.baseSet) (v : F) : (e.linearEquivAt R b hb).symm v = e.symm b v :=
  rfl


/-- A fiberwise linear inverse to `e`. -/
protected def symmₗ (e : Trivialization F (π F E)) [e.IsLinear R] (b : B) : F →ₗ[R] E b :=
  e.toPretrivialization.symmₗ R b


theorem coe_symmₗ (e : Trivialization F (π F E)) [e.IsLinear R] (b : B) :
    ⇑(e.symmₗ R b) = e.symm b :=
  rfl


/-- A fiberwise linear map equal to `e` on `e.baseSet`. -/
protected def linearMapAt (e : Trivialization F (π F E)) [e.IsLinear R] (b : B) : E b →ₗ[R] F :=
  e.toPretrivialization.linearMapAt R b


open Classical in
theorem coe_linearMapAt (e : Trivialization F (π F E)) [e.IsLinear R] (b : B) :
    ⇑(e.linearMapAt R b) = fun y => if b ∈ e.baseSet then (e ⟨b, y⟩).2 else 0 :=
  e.toPretrivialization.coe_linearMapAt b


theorem coe_linearMapAt_of_mem (e : Trivialization F (π F E)) [e.IsLinear R] {b : B}
    (hb : b ∈ e.baseSet) : ⇑(e.linearMapAt R b) = fun y => (e ⟨b, y⟩).2 := by
  /-
    R : Type u_1
    B : Type u_2
    F : Type u_3
    E : B → Type u_4
    inst✝⁸ : Semiring R
    inst✝⁷ : TopologicalSpace F
    inst✝⁶ : TopologicalSpace B
    inst✝⁵ : TopologicalSpace (Bundle.TotalSpace F E)
    inst✝⁴ : AddCommMonoid F
    inst✝³ : Module R F
    inst✝² : (x : B) → AddCommMonoid (E x)
    inst✝¹ : (x : B) → Module R (E x)
    e : Trivialization F Bundle.TotalSpace.proj
    inst✝ : Trivialization.IsLinear R e
    b : B
    hb : Membership.mem e.baseSet b
    ⊢ Eq ⇑(Trivialization.linearMapAt R e b) fun y => (↑e { proj := b, snd := y }).2
  -/
  simp_rw [coe_linearMapAt, if_pos hb]
  /-
    🎉 no goals
  -/


open Classical in
theorem linearMapAt_apply (e : Trivialization F (π F E)) [e.IsLinear R] {b : B} (y : E b) :
    e.linearMapAt R b y = if b ∈ e.baseSet then (e ⟨b, y⟩).2 else 0 := by
  /-
    R : Type u_1
    B : Type u_2
    F : Type u_3
    E : B → Type u_4
    inst✝⁸ : Semiring R
    inst✝⁷ : TopologicalSpace F
    inst✝⁶ : TopologicalSpace B
    inst✝⁵ : TopologicalSpace (Bundle.TotalSpace F E)
    inst✝⁴ : AddCommMonoid F
    inst✝³ : Module R F
    inst✝² : (x : B) → AddCommMonoid (E x)
    inst✝¹ : (x : B) → Module R (E x)
    e : Trivialization F Bundle.TotalSpace.proj
    inst✝ : Trivialization.IsLinear R e
    b : B
    y : E b
    ⊢ Eq ((Trivialization.linearMapAt R e b) y) (ite (Membership.mem e.baseSet b)  …
  -/
  rw [coe_linearMapAt]
  /-
    🎉 no goals
  -/


theorem linearMapAt_def_of_mem (e : Trivialization F (π F E)) [e.IsLinear R] {b : B}
    (hb : b ∈ e.baseSet) : e.linearMapAt R b = e.linearEquivAt R b hb :=
  dif_pos hb


theorem linearMapAt_def_of_not_mem (e : Trivialization F (π F E)) [e.IsLinear R] {b : B}
    (hb : b ∉ e.baseSet) : e.linearMapAt R b = 0 :=
  dif_neg hb


theorem symmₗ_linearMapAt (e : Trivialization F (π F E)) [e.IsLinear R] {b : B} (hb : b ∈ e.baseSet)
    (y : E b) : e.symmₗ R b (e.linearMapAt R b y) = y :=
  e.toPretrivialization.symmₗ_linearMapAt hb y


theorem linearMapAt_symmₗ (e : Trivialization F (π F E)) [e.IsLinear R] {b : B} (hb : b ∈ e.baseSet)
    (y : F) : e.linearMapAt R b (e.symmₗ R b y) = y :=
  e.toPretrivialization.linearMapAt_symmₗ hb y


open Classical in
/-- A coordinate change function between two trivializations, as a continuous linear equivalence.
  Defined to be the identity when `b` does not lie in the base set of both trivializations. -/
def coordChangeL (e e' : Trivialization F (π F E)) [e.IsLinear R] [e'.IsLinear R] (b : B) :
    F ≃L[R] F :=
  { toLinearEquiv := if hb : b ∈ e.baseSet ∩ e'.baseSet
      then (e.linearEquivAt R b (hb.1 : _)).symm.trans (e'.linearEquivAt R b hb.2)
      else LinearEquiv.refl R F
    continuous_toFun := by
      /-
        R : Type u_1
        B : Type u_2
        F : Type u_3
        E : B → Type u_4
        inst✝⁹ : Semiring R
        inst✝⁸ : TopologicalSpace F
        inst✝⁷ : TopologicalSpace B
        inst✝⁶ : TopologicalSpace (Bundle.TotalSpace F E)
        e✝ : Trivialization F Bundle.TotalSpace.proj
        x : Bundle.TotalSpace F E
        b✝ : B
        y : E b✝
        inst✝⁵ : AddCommMonoid F
        inst✝⁴ : Module R F
        inst✝³ : (x : B) → AddCommMonoid (E x)
        inst✝² : (x : B) → Module R (E x)
        e e' : Trivialization F Bundle.TotalSpace.proj
        inst✝¹ : Trivialization.IsLinear R e
        inst✝ : Trivialization.IsLinear R e'
        b : B
        ⊢ Continuous (↑(dite (Membership.mem (Inter.inter e.baseSet e'.baseSet) b) (fu …
      -/
      by_cases hb : b ∈ e.baseSet ∩ e'.baseSet
        /-
          case pos
          R : Type u_1
          B : Type u_2
          F : Type u_3
          E : B → Type u_4
          inst✝⁹ : Semiring R
          inst✝⁸ : TopologicalSpace F
          inst✝⁷ : TopologicalSpace B
          inst✝⁶ : TopologicalSpace (Bundle.TotalSpace F E)
          e✝ : Trivialization F Bundle.TotalSpace.proj
          x : Bundle.TotalSpace F E
          b✝ : B
          y : E b✝
          inst✝⁵ : AddCommMonoid F
          inst✝⁴ : Module R F
          inst✝³ : (x : B) → AddCommMonoid (E x)
          inst✝² : (x : B) → Module R (E x)
          e e' : Trivialization F Bundle.TotalSpace.proj
          inst✝¹ : Trivialization.IsLinear R e
          inst✝ : Trivialization.IsLinear R e'
          b : B
          hb : Membership.mem (Inter.inter e.baseSet e'.baseSet) b
          ⊢ Continuous (↑(dite (Membership.mem (Inter.inter e.baseSet e'.baseSet) b) (fu …
        -/
      · rw [dif_pos hb]
        /-
          case pos
          R : Type u_1
          B : Type u_2
          F : Type u_3
          E : B → Type u_4
          inst✝⁹ : Semiring R
          inst✝⁸ : TopologicalSpace F
          inst✝⁷ : TopologicalSpace B
          inst✝⁶ : TopologicalSpace (Bundle.TotalSpace F E)
          e✝ : Trivialization F Bundle.TotalSpace.proj
          x : Bundle.TotalSpace F E
          b✝ : B
          y : E b✝
          inst✝⁵ : AddCommMonoid F
          inst✝⁴ : Module R F
          inst✝³ : (x : B) → AddCommMonoid (E x)
          inst✝² : (x : B) → Module R (E x)
          e e' : Trivialization F Bundle.TotalSpace.proj
          inst✝¹ : Trivialization.IsLinear R e
          inst✝ : Trivialization.IsLinear R e'
          b : B
          hb : Membership.mem (Inter.inter e.baseSet e'.baseSet) b
          ⊢ Continuous (↑((Trivialization.linearEquivAt R e b ⋯).symm.trans (Trivializat …
        -/
        refine (e'.continuousOn.comp_continuous ?_ ?_).snd
        · exact e.continuousOn_symm.comp_continuous (Continuous.Prod.mk b) fun y =>
            mk_mem_prod hb.1 (mem_univ y)
          /-
            case pos.refine_2
            R : Type u_1
            B : Type u_2
            F : Type u_3
            E : B → Type u_4
            inst✝⁹ : Semiring R
            inst✝⁸ : TopologicalSpace F
            inst✝⁷ : TopologicalSpace B
            inst✝⁶ : TopologicalSpace (Bundle.TotalSpace F E)
            e✝ : Trivialization F Bundle.TotalSpace.proj
            x : Bundle.TotalSpace F E
            b✝ : B
            y : E b✝
            inst✝⁵ : AddCommMonoid F
            inst✝⁴ : Module R F
            inst✝³ : (x : B) → AddCommMonoid (E x)
            inst✝² : (x : B) → Module R (E x)
            e e' : Trivialization F Bundle.TotalSpace.proj
            inst✝¹ : Trivialization.IsLinear R e
            inst✝ : Trivialization.IsLinear R e'
            b : B
            hb : Membership.mem (Inter.inter e.baseSet e'.baseSet) b
            ⊢ ∀ (x : F), Membership.mem e'.source { proj := b, snd := ↑(Trivialization.lin …
          -/
        · exact fun y => e'.mem_source.mpr hb.2
          /-
            🎉 no goals
          -/
        /-
          case neg
          R : Type u_1
          B : Type u_2
          F : Type u_3
          E : B → Type u_4
          inst✝⁹ : Semiring R
          inst✝⁸ : TopologicalSpace F
          inst✝⁷ : TopologicalSpace B
          inst✝⁶ : TopologicalSpace (Bundle.TotalSpace F E)
          e✝ : Trivialization F Bundle.TotalSpace.proj
          x : Bundle.TotalSpace F E
          b✝ : B
          y : E b✝
          inst✝⁵ : AddCommMonoid F
          inst✝⁴ : Module R F
          inst✝³ : (x : B) → AddCommMonoid (E x)
          inst✝² : (x : B) → Module R (E x)
          e e' : Trivialization F Bundle.TotalSpace.proj
          inst✝¹ : Trivialization.IsLinear R e
          inst✝ : Trivialization.IsLinear R e'
          b : B
          hb : Not (Membership.mem (Inter.inter e.baseSet e'.baseSet) b)
          ⊢ Continuous (↑(dite (Membership.mem (Inter.inter e.baseSet e'.baseSet) b) (fu …
        -/
      · rw [dif_neg hb]
        /-
          case neg
          R : Type u_1
          B : Type u_2
          F : Type u_3
          E : B → Type u_4
          inst✝⁹ : Semiring R
          inst✝⁸ : TopologicalSpace F
          inst✝⁷ : TopologicalSpace B
          inst✝⁶ : TopologicalSpace (Bundle.TotalSpace F E)
          e✝ : Trivialization F Bundle.TotalSpace.proj
          x : Bundle.TotalSpace F E
          b✝ : B
          y : E b✝
          inst✝⁵ : AddCommMonoid F
          inst✝⁴ : Module R F
          inst✝³ : (x : B) → AddCommMonoid (E x)
          inst✝² : (x : B) → Module R (E x)
          e e' : Trivialization F Bundle.TotalSpace.proj
          inst✝¹ : Trivialization.IsLinear R e
          inst✝ : Trivialization.IsLinear R e'
          b : B
          hb : Not (Membership.mem (Inter.inter e.baseSet e'.baseSet) b)
          ⊢ Continuous (↑(LinearEquiv.refl R F)).toFun
        -/
        exact continuous_id
        /-
          🎉 no goals
        -/
    continuous_invFun := by
      /-
        R : Type u_1
        B : Type u_2
        F : Type u_3
        E : B → Type u_4
        inst✝⁹ : Semiring R
        inst✝⁸ : TopologicalSpace F
        inst✝⁷ : TopologicalSpace B
        inst✝⁶ : TopologicalSpace (Bundle.TotalSpace F E)
        e✝ : Trivialization F Bundle.TotalSpace.proj
        x : Bundle.TotalSpace F E
        b✝ : B
        y : E b✝
        inst✝⁵ : AddCommMonoid F
        inst✝⁴ : Module R F
        inst✝³ : (x : B) → AddCommMonoid (E x)
        inst✝² : (x : B) → Module R (E x)
        e e' : Trivialization F Bundle.TotalSpace.proj
        inst✝¹ : Trivialization.IsLinear R e
        inst✝ : Trivialization.IsLinear R e'
        b : B
        ⊢ Continuous (dite (Membership.mem (Inter.inter e.baseSet e'.baseSet) b) (fun  …
      -/
      by_cases hb : b ∈ e.baseSet ∩ e'.baseSet
        /-
          case pos
          R : Type u_1
          B : Type u_2
          F : Type u_3
          E : B → Type u_4
          inst✝⁹ : Semiring R
          inst✝⁸ : TopologicalSpace F
          inst✝⁷ : TopologicalSpace B
          inst✝⁶ : TopologicalSpace (Bundle.TotalSpace F E)
          e✝ : Trivialization F Bundle.TotalSpace.proj
          x : Bundle.TotalSpace F E
          b✝ : B
          y : E b✝
          inst✝⁵ : AddCommMonoid F
          inst✝⁴ : Module R F
          inst✝³ : (x : B) → AddCommMonoid (E x)
          inst✝² : (x : B) → Module R (E x)
          e e' : Trivialization F Bundle.TotalSpace.proj
          inst✝¹ : Trivialization.IsLinear R e
          inst✝ : Trivialization.IsLinear R e'
          b : B
          hb : Membership.mem (Inter.inter e.baseSet e'.baseSet) b
          ⊢ Continuous (dite (Membership.mem (Inter.inter e.baseSet e'.baseSet) b) (fun  …
        -/
      · rw [dif_pos hb]
        /-
          case pos
          R : Type u_1
          B : Type u_2
          F : Type u_3
          E : B → Type u_4
          inst✝⁹ : Semiring R
          inst✝⁸ : TopologicalSpace F
          inst✝⁷ : TopologicalSpace B
          inst✝⁶ : TopologicalSpace (Bundle.TotalSpace F E)
          e✝ : Trivialization F Bundle.TotalSpace.proj
          x : Bundle.TotalSpace F E
          b✝ : B
          y : E b✝
          inst✝⁵ : AddCommMonoid F
          inst✝⁴ : Module R F
          inst✝³ : (x : B) → AddCommMonoid (E x)
          inst✝² : (x : B) → Module R (E x)
          e e' : Trivialization F Bundle.TotalSpace.proj
          inst✝¹ : Trivialization.IsLinear R e
          inst✝ : Trivialization.IsLinear R e'
          b : B
          hb : Membership.mem (Inter.inter e.baseSet e'.baseSet) b
          ⊢ Continuous ((Trivialization.linearEquivAt R e b ⋯).symm.trans (Trivializatio …
        -/
        refine (e.continuousOn.comp_continuous ?_ ?_).snd
        · exact e'.continuousOn_symm.comp_continuous (Continuous.Prod.mk b) fun y =>
            mk_mem_prod hb.2 (mem_univ y)
        /-
          case pos.refine_2
          R : Type u_1
          B : Type u_2
          F : Type u_3
          E : B → Type u_4
          inst✝⁹ : Semiring R
          inst✝⁸ : TopologicalSpace F
          inst✝⁷ : TopologicalSpace B
          inst✝⁶ : TopologicalSpace (Bundle.TotalSpace F E)
          e✝ : Trivialization F Bundle.TotalSpace.proj
          x : Bundle.TotalSpace F E
          b✝ : B
          y : E b✝
          inst✝⁵ : AddCommMonoid F
          inst✝⁴ : Module R F
          inst✝³ : (x : B) → AddCommMonoid (E x)
          inst✝² : (x : B) → Module R (E x)
          e e' : Trivialization F Bundle.TotalSpace.proj
          inst✝¹ : Trivialization.IsLinear R e
          inst✝ : Trivialization.IsLinear R e'
          b : B
          hb : Membership.mem (Inter.inter e.baseSet e'.baseSet) b
          ⊢ ∀ (x : F), Membership.mem e.source { proj := b, snd := (Trivialization.linea …
        -/
        exact fun y => e.mem_source.mpr hb.1
        /-
          🎉 no goals
        -/
        /-
          case neg
          R : Type u_1
          B : Type u_2
          F : Type u_3
          E : B → Type u_4
          inst✝⁹ : Semiring R
          inst✝⁸ : TopologicalSpace F
          inst✝⁷ : TopologicalSpace B
          inst✝⁶ : TopologicalSpace (Bundle.TotalSpace F E)
          e✝ : Trivialization F Bundle.TotalSpace.proj
          x : Bundle.TotalSpace F E
          b✝ : B
          y : E b✝
          inst✝⁵ : AddCommMonoid F
          inst✝⁴ : Module R F
          inst✝³ : (x : B) → AddCommMonoid (E x)
          inst✝² : (x : B) → Module R (E x)
          e e' : Trivialization F Bundle.TotalSpace.proj
          inst✝¹ : Trivialization.IsLinear R e
          inst✝ : Trivialization.IsLinear R e'
          b : B
          hb : Not (Membership.mem (Inter.inter e.baseSet e'.baseSet) b)
          ⊢ Continuous (dite (Membership.mem (Inter.inter e.baseSet e'.baseSet) b) (fun  …
        -/
      · rw [dif_neg hb]
        /-
          case neg
          R : Type u_1
          B : Type u_2
          F : Type u_3
          E : B → Type u_4
          inst✝⁹ : Semiring R
          inst✝⁸ : TopologicalSpace F
          inst✝⁷ : TopologicalSpace B
          inst✝⁶ : TopologicalSpace (Bundle.TotalSpace F E)
          e✝ : Trivialization F Bundle.TotalSpace.proj
          x : Bundle.TotalSpace F E
          b✝ : B
          y : E b✝
          inst✝⁵ : AddCommMonoid F
          inst✝⁴ : Module R F
          inst✝³ : (x : B) → AddCommMonoid (E x)
          inst✝² : (x : B) → Module R (E x)
          e e' : Trivialization F Bundle.TotalSpace.proj
          inst✝¹ : Trivialization.IsLinear R e
          inst✝ : Trivialization.IsLinear R e'
          b : B
          hb : Not (Membership.mem (Inter.inter e.baseSet e'.baseSet) b)
          ⊢ Continuous (LinearEquiv.refl R F).invFun
        -/
        exact continuous_id }
        /-
          🎉 no goals
        -/


theorem coe_coordChangeL (e e' : Trivialization F (π F E)) [e.IsLinear R] [e'.IsLinear R] {b : B}
    (hb : b ∈ e.baseSet ∩ e'.baseSet) :
    ⇑(coordChangeL R e e' b) = (e.linearEquivAt R b hb.1).symm.trans (e'.linearEquivAt R b hb.2) :=
  congr_arg (fun f : F ≃ₗ[R] F ↦ ⇑f) (dif_pos hb)


theorem coe_coordChangeL' (e e' : Trivialization F (π F E)) [e.IsLinear R] [e'.IsLinear R] {b : B}
    (hb : b ∈ e.baseSet ∩ e'.baseSet) :
    (coordChangeL R e e' b).toLinearEquiv =
      (e.linearEquivAt R b hb.1).symm.trans (e'.linearEquivAt R b hb.2) :=
  LinearEquiv.coe_injective (coe_coordChangeL _ _ hb)


theorem symm_coordChangeL (e e' : Trivialization F (π F E)) [e.IsLinear R] [e'.IsLinear R] {b : B}
    (hb : b ∈ e'.baseSet ∩ e.baseSet) : (e.coordChangeL R e' b).symm = e'.coordChangeL R e b := by
  /-
    R : Type u_1
    B : Type u_2
    F : Type u_3
    E : B → Type u_4
    inst✝⁹ : Semiring R
    inst✝⁸ : TopologicalSpace F
    inst✝⁷ : TopologicalSpace B
    inst✝⁶ : TopologicalSpace (Bundle.TotalSpace F E)
    inst✝⁵ : AddCommMonoid F
    inst✝⁴ : Module R F
    inst✝³ : (x : B) → AddCommMonoid (E x)
    inst✝² : (x : B) → Module R (E x)
    e e' : Trivialization F Bundle.TotalSpace.proj
    inst✝¹ : Trivialization.IsLinear R e
    inst✝ : Trivialization.IsLinear R e'
    b : B
    hb : Membership.mem (Inter.inter e'.baseSet e.baseSet) b
    ⊢ Eq (Trivialization.coordChangeL R e e' b).symm (Trivialization.coordChangeL  …
  -/
  apply ContinuousLinearEquiv.toLinearEquiv_injective
  rw [coe_coordChangeL' e' e hb, (coordChangeL R e e' b).symm_toLinearEquiv,
    coe_coordChangeL' e e' hb.symm, LinearEquiv.trans_symm, LinearEquiv.symm_symm]


theorem coordChangeL_apply (e e' : Trivialization F (π F E)) [e.IsLinear R] [e'.IsLinear R] {b : B}
    (hb : b ∈ e.baseSet ∩ e'.baseSet) (y : F) :
    coordChangeL R e e' b y = (e' ⟨b, e.symm b y⟩).2 :=
  congr_fun (coe_coordChangeL e e' hb) y


theorem mk_coordChangeL (e e' : Trivialization F (π F E)) [e.IsLinear R] [e'.IsLinear R] {b : B}
    (hb : b ∈ e.baseSet ∩ e'.baseSet) (y : F) :
    (b, coordChangeL R e e' b y) = e' ⟨b, e.symm b y⟩ := by
  /-
    R : Type u_1
    B : Type u_2
    F : Type u_3
    E : B → Type u_4
    inst✝⁹ : Semiring R
    inst✝⁸ : TopologicalSpace F
    inst✝⁷ : TopologicalSpace B
    inst✝⁶ : TopologicalSpace (Bundle.TotalSpace F E)
    inst✝⁵ : AddCommMonoid F
    inst✝⁴ : Module R F
    inst✝³ : (x : B) → AddCommMonoid (E x)
    inst✝² : (x : B) → Module R (E x)
    e e' : Trivialization F Bundle.TotalSpace.proj
    inst✝¹ : Trivialization.IsLinear R e
    inst✝ : Trivialization.IsLinear R e'
    b : B
    hb : Membership.mem (Inter.inter e.baseSet e'.baseSet) b
    y : F
    ⊢ Eq { fst := b, snd := (Trivialization.coordChangeL R e e' b) y } (↑e' { proj …
  -/
  ext
    /-
      case fst
      R : Type u_1
      B : Type u_2
      F : Type u_3
      E : B → Type u_4
      inst✝⁹ : Semiring R
      inst✝⁸ : TopologicalSpace F
      inst✝⁷ : TopologicalSpace B
      inst✝⁶ : TopologicalSpace (Bundle.TotalSpace F E)
      inst✝⁵ : AddCommMonoid F
      inst✝⁴ : Module R F
      inst✝³ : (x : B) → AddCommMonoid (E x)
      inst✝² : (x : B) → Module R (E x)
      e e' : Trivialization F Bundle.TotalSpace.proj
      inst✝¹ : Trivialization.IsLinear R e
      inst✝ : Trivialization.IsLinear R e'
      b : B
      hb : Membership.mem (Inter.inter e.baseSet e'.baseSet) b
      y : F
      ⊢ Eq { fst := b, snd := (Trivialization.coordChangeL R e e' b) y }.1 (↑e' { pr …
    -/
  · rw [e.mk_symm hb.1 y, e'.coe_fst', e.proj_symm_apply' hb.1]
    /-
      case fst
      R : Type u_1
      B : Type u_2
      F : Type u_3
      E : B → Type u_4
      inst✝⁹ : Semiring R
      inst✝⁸ : TopologicalSpace F
      inst✝⁷ : TopologicalSpace B
      inst✝⁶ : TopologicalSpace (Bundle.TotalSpace F E)
      inst✝⁵ : AddCommMonoid F
      inst✝⁴ : Module R F
      inst✝³ : (x : B) → AddCommMonoid (E x)
      inst✝² : (x : B) → Module R (E x)
      e e' : Trivialization F Bundle.TotalSpace.proj
      inst✝¹ : Trivialization.IsLinear R e
      inst✝ : Trivialization.IsLinear R e'
      b : B
      hb : Membership.mem (Inter.inter e.baseSet e'.baseSet) b
      y : F
      ⊢ Membership.mem e'.baseSet (↑e.symm { fst := b, snd := y }).proj
    -/
    rw [e.proj_symm_apply' hb.1]
    /-
      case fst
      R : Type u_1
      B : Type u_2
      F : Type u_3
      E : B → Type u_4
      inst✝⁹ : Semiring R
      inst✝⁸ : TopologicalSpace F
      inst✝⁷ : TopologicalSpace B
      inst✝⁶ : TopologicalSpace (Bundle.TotalSpace F E)
      inst✝⁵ : AddCommMonoid F
      inst✝⁴ : Module R F
      inst✝³ : (x : B) → AddCommMonoid (E x)
      inst✝² : (x : B) → Module R (E x)
      e e' : Trivialization F Bundle.TotalSpace.proj
      inst✝¹ : Trivialization.IsLinear R e
      inst✝ : Trivialization.IsLinear R e'
      b : B
      hb : Membership.mem (Inter.inter e.baseSet e'.baseSet) b
      y : F
      ⊢ Membership.mem e'.baseSet b
    -/
    exact hb.2
    /-
      🎉 no goals
    -/
    /-
      case snd
      R : Type u_1
      B : Type u_2
      F : Type u_3
      E : B → Type u_4
      inst✝⁹ : Semiring R
      inst✝⁸ : TopologicalSpace F
      inst✝⁷ : TopologicalSpace B
      inst✝⁶ : TopologicalSpace (Bundle.TotalSpace F E)
      inst✝⁵ : AddCommMonoid F
      inst✝⁴ : Module R F
      inst✝³ : (x : B) → AddCommMonoid (E x)
      inst✝² : (x : B) → Module R (E x)
      e e' : Trivialization F Bundle.TotalSpace.proj
      inst✝¹ : Trivialization.IsLinear R e
      inst✝ : Trivialization.IsLinear R e'
      b : B
      hb : Membership.mem (Inter.inter e.baseSet e'.baseSet) b
      y : F
      ⊢ Eq { fst := b, snd := (Trivialization.coordChangeL R e e' b) y }.2 (↑e' { pr …
    -/
  · exact e.coordChangeL_apply e' hb y
    /-
      🎉 no goals
    -/


theorem apply_symm_apply_eq_coordChangeL (e e' : Trivialization F (π F E)) [e.IsLinear R]
    [e'.IsLinear R] {b : B} (hb : b ∈ e.baseSet ∩ e'.baseSet) (v : F) :
    e' (e.toPartialHomeomorph.symm (b, v)) = (b, e.coordChangeL R e' b v) := by
  /-
    R : Type u_1
    B : Type u_2
    F : Type u_3
    E : B → Type u_4
    inst✝⁹ : Semiring R
    inst✝⁸ : TopologicalSpace F
    inst✝⁷ : TopologicalSpace B
    inst✝⁶ : TopologicalSpace (Bundle.TotalSpace F E)
    inst✝⁵ : AddCommMonoid F
    inst✝⁴ : Module R F
    inst✝³ : (x : B) → AddCommMonoid (E x)
    inst✝² : (x : B) → Module R (E x)
    e e' : Trivialization F Bundle.TotalSpace.proj
    inst✝¹ : Trivialization.IsLinear R e
    inst✝ : Trivialization.IsLinear R e'
    b : B
    hb : Membership.mem (Inter.inter e.baseSet e'.baseSet) b
    v : F
    ⊢ Eq (↑e' (↑e.symm { fst := b, snd := v })) { fst := b, snd := (Trivialization …
  -/
  rw [e.mk_coordChangeL e' hb, e.mk_symm hb.1]
  /-
    🎉 no goals
  -/


/-- A version of `Trivialization.coordChangeL_apply` that fully unfolds `coordChange`. The
right-hand side is ugly, but has good definitional properties for specifically defined
trivializations. -/
theorem coordChangeL_apply' (e e' : Trivialization F (π F E)) [e.IsLinear R] [e'.IsLinear R] {b : B}
    (hb : b ∈ e.baseSet ∩ e'.baseSet) (y : F) :
    coordChangeL R e e' b y = (e' (e.toPartialHomeomorph.symm (b, y))).2 := by
  /-
    R : Type u_1
    B : Type u_2
    F : Type u_3
    E : B → Type u_4
    inst✝⁹ : Semiring R
    inst✝⁸ : TopologicalSpace F
    inst✝⁷ : TopologicalSpace B
    inst✝⁶ : TopologicalSpace (Bundle.TotalSpace F E)
    inst✝⁵ : AddCommMonoid F
    inst✝⁴ : Module R F
    inst✝³ : (x : B) → AddCommMonoid (E x)
    inst✝² : (x : B) → Module R (E x)
    e e' : Trivialization F Bundle.TotalSpace.proj
    inst✝¹ : Trivialization.IsLinear R e
    inst✝ : Trivialization.IsLinear R e'
    b : B
    hb : Membership.mem (Inter.inter e.baseSet e'.baseSet) b
    y : F
    ⊢ Eq ((Trivialization.coordChangeL R e e' b) y) (↑e' (↑e.symm { fst := b, snd  …
  -/
  rw [e.coordChangeL_apply e' hb, e.mk_symm hb.1]
  /-
    🎉 no goals
  -/


theorem coordChangeL_symm_apply (e e' : Trivialization F (π F E)) [e.IsLinear R] [e'.IsLinear R]
    {b : B} (hb : b ∈ e.baseSet ∩ e'.baseSet) :
    ⇑(coordChangeL R e e' b).symm =
      (e'.linearEquivAt R b hb.2).symm.trans (e.linearEquivAt R b hb.1) :=
  congr_arg LinearEquiv.invFun (dif_pos hb)


/-- The zero section of a vector bundle -/
def zeroSection [∀ x, Zero (E x)] : B → TotalSpace F E := (⟨·, 0⟩)


@[simp, mfld_simps]
theorem zeroSection_proj [∀ x, Zero (E x)] (x : B) : (zeroSection F E x).proj = x :=
  rfl


@[simp, mfld_simps]
theorem zeroSection_snd [∀ x, Zero (E x)] (x : B) : (zeroSection F E x).2 = 0 :=
  rfl


/-- The space `Bundle.TotalSpace F E` (for `E : B → Type*` such that each `E x` is a topological
vector space) has a topological vector space structure with fiber `F` (denoted with
`VectorBundle R F E`) if around every point there is a fiber bundle trivialization which is linear
in the fibers. -/
class VectorBundle : Prop where
  trivialization_linear' : ∀ (e : Trivialization F (π F E)) [MemTrivializationAtlas e], e.IsLinear R
  continuousOn_coordChange' :
    ∀ (e e' : Trivialization F (π F E)) [MemTrivializationAtlas e] [MemTrivializationAtlas e'],
      ContinuousOn (fun b => Trivialization.coordChangeL R e e' b : B → F →L[R] F)
        (e.baseSet ∩ e'.baseSet)


instance (priority := 100) trivialization_linear [VectorBundle R F E] (e : Trivialization F (π F E))
    [MemTrivializationAtlas e] : e.IsLinear R :=
  VectorBundle.trivialization_linear' e


theorem continuousOn_coordChange [VectorBundle R F E] (e e' : Trivialization F (π F E))
    [MemTrivializationAtlas e] [MemTrivializationAtlas e'] :
    ContinuousOn (fun b => Trivialization.coordChangeL R e e' b : B → F →L[R] F)
      (e.baseSet ∩ e'.baseSet) :=
  VectorBundle.continuousOn_coordChange' e e'


/-- Forward map of `Trivialization.continuousLinearEquivAt` (only propositionally equal),
  defined everywhere (`0` outside domain). -/
@[simps (config := .asFn) apply]
def continuousLinearMapAt (e : Trivialization F (π F E)) [e.IsLinear R] (b : B) : E b →L[R] F :=
  { e.linearMapAt R b with
    toFun := e.linearMapAt R b -- given explicitly to help `simps`
    cont := by
      /-
        R : Type u_1
        B : Type u_2
        F : Type u_3
        E : B → Type u_4
        inst✝⁹ : NontriviallyNormedField R
        inst✝⁸ : (x : B) → AddCommMonoid (E x)
        inst✝⁷ : (x : B) → Module R (E x)
        inst✝⁶ : NormedAddCommGroup F
        inst✝⁵ : NormedSpace R F
        inst✝⁴ : TopologicalSpace B
        inst✝³ : TopologicalSpace (Bundle.TotalSpace F E)
        inst✝² : (x : B) → TopologicalSpace (E x)
        inst✝¹ : FiberBundle F E
        e : Trivialization F Bundle.TotalSpace.proj
        inst✝ : Trivialization.IsLinear R e
        b : B
        ⊢ Continuous { toFun := ⇑(Trivialization.linearMapAt R e b), map_add' := ⋯, ma …
      -/
      dsimp
      /-
        R : Type u_1
        B : Type u_2
        F : Type u_3
        E : B → Type u_4
        inst✝⁹ : NontriviallyNormedField R
        inst✝⁸ : (x : B) → AddCommMonoid (E x)
        inst✝⁷ : (x : B) → Module R (E x)
        inst✝⁶ : NormedAddCommGroup F
        inst✝⁵ : NormedSpace R F
        inst✝⁴ : TopologicalSpace B
        inst✝³ : TopologicalSpace (Bundle.TotalSpace F E)
        inst✝² : (x : B) → TopologicalSpace (E x)
        inst✝¹ : FiberBundle F E
        e : Trivialization F Bundle.TotalSpace.proj
        inst✝ : Trivialization.IsLinear R e
        b : B
        ⊢ Continuous ⇑(Trivialization.linearMapAt R e b)
      -/
      rw [e.coe_linearMapAt b]
      classical
      refine continuous_if_const _ (fun hb => ?_) fun _ => continuous_zero
      exact (e.continuousOn.comp_continuous (FiberBundle.totalSpaceMk_isInducing F E b).continuous
        fun x => e.mem_source.mpr hb).snd }


/-- Backwards map of `Trivialization.continuousLinearEquivAt`, defined everywhere. -/
@[simps (config := .asFn) apply]
def symmL (e : Trivialization F (π F E)) [e.IsLinear R] (b : B) : F →L[R] E b :=
  { e.symmₗ R b with
    toFun := e.symm b -- given explicitly to help `simps`
    cont := by
      /-
        R : Type u_1
        B : Type u_2
        F : Type u_3
        E : B → Type u_4
        inst✝⁹ : NontriviallyNormedField R
        inst✝⁸ : (x : B) → AddCommMonoid (E x)
        inst✝⁷ : (x : B) → Module R (E x)
        inst✝⁶ : NormedAddCommGroup F
        inst✝⁵ : NormedSpace R F
        inst✝⁴ : TopologicalSpace B
        inst✝³ : TopologicalSpace (Bundle.TotalSpace F E)
        inst✝² : (x : B) → TopologicalSpace (E x)
        inst✝¹ : FiberBundle F E
        e : Trivialization F Bundle.TotalSpace.proj
        inst✝ : Trivialization.IsLinear R e
        b : B
        ⊢ Continuous { toFun := e.symm b, map_add' := ⋯, map_smul' := ⋯ }.toFun
      -/
      by_cases hb : b ∈ e.baseSet
        /-
          case pos
          R : Type u_1
          B : Type u_2
          F : Type u_3
          E : B → Type u_4
          inst✝⁹ : NontriviallyNormedField R
          inst✝⁸ : (x : B) → AddCommMonoid (E x)
          inst✝⁷ : (x : B) → Module R (E x)
          inst✝⁶ : NormedAddCommGroup F
          inst✝⁵ : NormedSpace R F
          inst✝⁴ : TopologicalSpace B
          inst✝³ : TopologicalSpace (Bundle.TotalSpace F E)
          inst✝² : (x : B) → TopologicalSpace (E x)
          inst✝¹ : FiberBundle F E
          e : Trivialization F Bundle.TotalSpace.proj
          inst✝ : Trivialization.IsLinear R e
          b : B
          hb : Membership.mem e.baseSet b
          ⊢ Continuous { toFun := e.symm b, map_add' := ⋯, map_smul' := ⋯ }.toFun
        -/
      · rw [(FiberBundle.totalSpaceMk_isInducing F E b).continuous_iff]
        exact e.continuousOn_symm.comp_continuous (continuous_const.prod_mk continuous_id) fun x ↦
          mk_mem_prod hb (mem_univ x)
        /-
          case neg
          R : Type u_1
          B : Type u_2
          F : Type u_3
          E : B → Type u_4
          inst✝⁹ : NontriviallyNormedField R
          inst✝⁸ : (x : B) → AddCommMonoid (E x)
          inst✝⁷ : (x : B) → Module R (E x)
          inst✝⁶ : NormedAddCommGroup F
          inst✝⁵ : NormedSpace R F
          inst✝⁴ : TopologicalSpace B
          inst✝³ : TopologicalSpace (Bundle.TotalSpace F E)
          inst✝² : (x : B) → TopologicalSpace (E x)
          inst✝¹ : FiberBundle F E
          e : Trivialization F Bundle.TotalSpace.proj
          inst✝ : Trivialization.IsLinear R e
          b : B
          hb : Not (Membership.mem e.baseSet b)
          ⊢ Continuous { toFun := e.symm b, map_add' := ⋯, map_smul' := ⋯ }.toFun
        -/
      · refine continuous_zero.congr fun x => (e.symm_apply_of_not_mem hb x).symm }
        /-
          🎉 no goals
        -/


theorem symmL_continuousLinearMapAt (e : Trivialization F (π F E)) [e.IsLinear R] {b : B}
    (hb : b ∈ e.baseSet) (y : E b) : e.symmL R b (e.continuousLinearMapAt R b y) = y :=
  e.symmₗ_linearMapAt hb y


theorem continuousLinearMapAt_symmL (e : Trivialization F (π F E)) [e.IsLinear R] {b : B}
    (hb : b ∈ e.baseSet) (y : F) : e.continuousLinearMapAt R b (e.symmL R b y) = y :=
  e.linearMapAt_symmₗ hb y


/-- In a vector bundle, a trivialization in the fiber (which is a priori only linear)
is in fact a continuous linear equiv between the fibers and the model fiber. -/
@[simps (config := .asFn) apply symm_apply]
def continuousLinearEquivAt (e : Trivialization F (π F E)) [e.IsLinear R] (b : B)
    (hb : b ∈ e.baseSet) : E b ≃L[R] F :=
  { e.toPretrivialization.linearEquivAt R b hb with
    toFun := fun y => (e ⟨b, y⟩).2 -- given explicitly to help `simps`
    invFun := e.symm b -- given explicitly to help `simps`
    continuous_toFun := (e.continuousOn.comp_continuous
      (FiberBundle.totalSpaceMk_isInducing F E b).continuous fun _ => e.mem_source.mpr hb).snd
    continuous_invFun := (e.symmL R b).continuous }


theorem coe_continuousLinearEquivAt_eq (e : Trivialization F (π F E)) [e.IsLinear R] {b : B}
    (hb : b ∈ e.baseSet) :
    (e.continuousLinearEquivAt R b hb : E b → F) = e.continuousLinearMapAt R b :=
  (e.coe_linearMapAt_of_mem hb).symm


theorem symm_continuousLinearEquivAt_eq (e : Trivialization F (π F E)) [e.IsLinear R] {b : B}
    (hb : b ∈ e.baseSet) : ((e.continuousLinearEquivAt R b hb).symm : F → E b) = e.symmL R b :=
  rfl


@[simp]
theorem continuousLinearEquivAt_apply' (e : Trivialization F (π F E)) [e.IsLinear R]
    (x : TotalSpace F E) (hx : x ∈ e.source) :
    e.continuousLinearEquivAt R x.proj (e.mem_source.1 hx) x.2 = (e x).2 := rfl


theorem apply_eq_prod_continuousLinearEquivAt (e : Trivialization F (π F E)) [e.IsLinear R] (b : B)
    (hb : b ∈ e.baseSet) (z : E b) : e ⟨b, z⟩ = (b, e.continuousLinearEquivAt R b hb z) := by
  /-
    R : Type u_1
    B : Type u_2
    F : Type u_3
    E : B → Type u_4
    inst✝⁹ : NontriviallyNormedField R
    inst✝⁸ : (x : B) → AddCommMonoid (E x)
    inst✝⁷ : (x : B) → Module R (E x)
    inst✝⁶ : NormedAddCommGroup F
    inst✝⁵ : NormedSpace R F
    inst✝⁴ : TopologicalSpace B
    inst✝³ : TopologicalSpace (Bundle.TotalSpace F E)
    inst✝² : (x : B) → TopologicalSpace (E x)
    inst✝¹ : FiberBundle F E
    e : Trivialization F Bundle.TotalSpace.proj
    inst✝ : Trivialization.IsLinear R e
    b : B
    hb : Membership.mem e.baseSet b
    z : E b
    ⊢ Eq (↑e { proj := b, snd := z }) { fst := b, snd := (Trivialization.continuou …
  -/
  ext
    /-
      case fst
      R : Type u_1
      B : Type u_2
      F : Type u_3
      E : B → Type u_4
      inst✝⁹ : NontriviallyNormedField R
      inst✝⁸ : (x : B) → AddCommMonoid (E x)
      inst✝⁷ : (x : B) → Module R (E x)
      inst✝⁶ : NormedAddCommGroup F
      inst✝⁵ : NormedSpace R F
      inst✝⁴ : TopologicalSpace B
      inst✝³ : TopologicalSpace (Bundle.TotalSpace F E)
      inst✝² : (x : B) → TopologicalSpace (E x)
      inst✝¹ : FiberBundle F E
      e : Trivialization F Bundle.TotalSpace.proj
      inst✝ : Trivialization.IsLinear R e
      b : B
      hb : Membership.mem e.baseSet b
      z : E b
      ⊢ Eq (↑e { proj := b, snd := z }).1 { fst := b, snd := (Trivialization.continu …
    -/
  · refine e.coe_fst ?_
    /-
      case fst
      R : Type u_1
      B : Type u_2
      F : Type u_3
      E : B → Type u_4
      inst✝⁹ : NontriviallyNormedField R
      inst✝⁸ : (x : B) → AddCommMonoid (E x)
      inst✝⁷ : (x : B) → Module R (E x)
      inst✝⁶ : NormedAddCommGroup F
      inst✝⁵ : NormedSpace R F
      inst✝⁴ : TopologicalSpace B
      inst✝³ : TopologicalSpace (Bundle.TotalSpace F E)
      inst✝² : (x : B) → TopologicalSpace (E x)
      inst✝¹ : FiberBundle F E
      e : Trivialization F Bundle.TotalSpace.proj
      inst✝ : Trivialization.IsLinear R e
      b : B
      hb : Membership.mem e.baseSet b
      z : E b
      ⊢ Membership.mem e.source { proj := b, snd := z }
    -/
    rw [e.source_eq]
    /-
      case fst
      R : Type u_1
      B : Type u_2
      F : Type u_3
      E : B → Type u_4
      inst✝⁹ : NontriviallyNormedField R
      inst✝⁸ : (x : B) → AddCommMonoid (E x)
      inst✝⁷ : (x : B) → Module R (E x)
      inst✝⁶ : NormedAddCommGroup F
      inst✝⁵ : NormedSpace R F
      inst✝⁴ : TopologicalSpace B
      inst✝³ : TopologicalSpace (Bundle.TotalSpace F E)
      inst✝² : (x : B) → TopologicalSpace (E x)
      inst✝¹ : FiberBundle F E
      e : Trivialization F Bundle.TotalSpace.proj
      inst✝ : Trivialization.IsLinear R e
      b : B
      hb : Membership.mem e.baseSet b
      z : E b
      ⊢ Membership.mem (Set.preimage Bundle.TotalSpace.proj e.baseSet) { proj := b,  …
    -/
    exact hb
    /-
      🎉 no goals
    -/
    /-
      case snd
      R : Type u_1
      B : Type u_2
      F : Type u_3
      E : B → Type u_4
      inst✝⁹ : NontriviallyNormedField R
      inst✝⁸ : (x : B) → AddCommMonoid (E x)
      inst✝⁷ : (x : B) → Module R (E x)
      inst✝⁶ : NormedAddCommGroup F
      inst✝⁵ : NormedSpace R F
      inst✝⁴ : TopologicalSpace B
      inst✝³ : TopologicalSpace (Bundle.TotalSpace F E)
      inst✝² : (x : B) → TopologicalSpace (E x)
      inst✝¹ : FiberBundle F E
      e : Trivialization F Bundle.TotalSpace.proj
      inst✝ : Trivialization.IsLinear R e
      b : B
      hb : Membership.mem e.baseSet b
      z : E b
      ⊢ Eq (↑e { proj := b, snd := z }).2 { fst := b, snd := (Trivialization.continu …
    -/
  · simp only [coe_coe, continuousLinearEquivAt_apply]
    /-
      🎉 no goals
    -/


protected theorem zeroSection (e : Trivialization F (π F E)) [e.IsLinear R] {x : B}
    (hx : x ∈ e.baseSet) : e (zeroSection F E x) = (x, 0) := by
  /-
    R : Type u_1
    B : Type u_2
    F : Type u_3
    E : B → Type u_4
    inst✝⁹ : NontriviallyNormedField R
    inst✝⁸ : (x : B) → AddCommMonoid (E x)
    inst✝⁷ : (x : B) → Module R (E x)
    inst✝⁶ : NormedAddCommGroup F
    inst✝⁵ : NormedSpace R F
    inst✝⁴ : TopologicalSpace B
    inst✝³ : TopologicalSpace (Bundle.TotalSpace F E)
    inst✝² : (x : B) → TopologicalSpace (E x)
    inst✝¹ : FiberBundle F E
    e : Trivialization F Bundle.TotalSpace.proj
    inst✝ : Trivialization.IsLinear R e
    x : B
    hx : Membership.mem e.baseSet x
    ⊢ Eq (↑e (Bundle.zeroSection F E x)) { fst := x, snd := 0 }
  -/
  simp_rw [zeroSection, e.apply_eq_prod_continuousLinearEquivAt R x hx 0, map_zero]
  /-
    🎉 no goals
  -/


theorem symm_apply_eq_mk_continuousLinearEquivAt_symm (e : Trivialization F (π F E)) [e.IsLinear R]
    (b : B) (hb : b ∈ e.baseSet) (z : F) :
    e.toPartialHomeomorph.symm ⟨b, z⟩ = ⟨b, (e.continuousLinearEquivAt R b hb).symm z⟩ := by
  have h : (b, z) ∈ e.target := by
    rw [e.target_eq]
    exact ⟨hb, mem_univ _⟩
  /-
    R : Type u_1
    B : Type u_2
    F : Type u_3
    E : B → Type u_4
    inst✝⁹ : NontriviallyNormedField R
    inst✝⁸ : (x : B) → AddCommMonoid (E x)
    inst✝⁷ : (x : B) → Module R (E x)
    inst✝⁶ : NormedAddCommGroup F
    inst✝⁵ : NormedSpace R F
    inst✝⁴ : TopologicalSpace B
    inst✝³ : TopologicalSpace (Bundle.TotalSpace F E)
    inst✝² : (x : B) → TopologicalSpace (E x)
    inst✝¹ : FiberBundle F E
    e : Trivialization F Bundle.TotalSpace.proj
    inst✝ : Trivialization.IsLinear R e
    b : B
    hb : Membership.mem e.baseSet b
    z : F
    h : Membership.mem e.target { fst := b, snd := z }
    ⊢ Eq (↑e.symm { fst := b, snd := z }) { proj := b, snd := (Trivialization.cont …
  -/
  apply e.injOn (e.map_target h)
    /-
      case a
      R : Type u_1
      B : Type u_2
      F : Type u_3
      E : B → Type u_4
      inst✝⁹ : NontriviallyNormedField R
      inst✝⁸ : (x : B) → AddCommMonoid (E x)
      inst✝⁷ : (x : B) → Module R (E x)
      inst✝⁶ : NormedAddCommGroup F
      inst✝⁵ : NormedSpace R F
      inst✝⁴ : TopologicalSpace B
      inst✝³ : TopologicalSpace (Bundle.TotalSpace F E)
      inst✝² : (x : B) → TopologicalSpace (E x)
      inst✝¹ : FiberBundle F E
      e : Trivialization F Bundle.TotalSpace.proj
      inst✝ : Trivialization.IsLinear R e
      b : B
      hb : Membership.mem e.baseSet b
      z : F
      h : Membership.mem e.target { fst := b, snd := z }
      ⊢ Membership.mem e.source { proj := b, snd := (Trivialization.continuousLinear …
    -/
  · simpa only [e.source_eq, mem_preimage]
    /-
      🎉 no goals
    -/
  · simp_rw [e.right_inv h, coe_coe, e.apply_eq_prod_continuousLinearEquivAt R b hb,
      ContinuousLinearEquiv.apply_symm_apply]


theorem comp_continuousLinearEquivAt_eq_coord_change (e e' : Trivialization F (π F E))
    [e.IsLinear R] [e'.IsLinear R] {b : B} (hb : b ∈ e.baseSet ∩ e'.baseSet) :
    (e.continuousLinearEquivAt R b hb.1).symm.trans (e'.continuousLinearEquivAt R b hb.2) =
      coordChangeL R e e' b := by
  /-
    R : Type u_1
    B : Type u_2
    F : Type u_3
    E : B → Type u_4
    inst✝¹⁰ : NontriviallyNormedField R
    inst✝⁹ : (x : B) → AddCommMonoid (E x)
    inst✝⁸ : (x : B) → Module R (E x)
    inst✝⁷ : NormedAddCommGroup F
    inst✝⁶ : NormedSpace R F
    inst✝⁵ : TopologicalSpace B
    inst✝⁴ : TopologicalSpace (Bundle.TotalSpace F E)
    inst✝³ : (x : B) → TopologicalSpace (E x)
    inst✝² : FiberBundle F E
    e e' : Trivialization F Bundle.TotalSpace.proj
    inst✝¹ : Trivialization.IsLinear R e
    inst✝ : Trivialization.IsLinear R e'
    b : B
    hb : Membership.mem (Inter.inter e.baseSet e'.baseSet) b
    ⊢ Eq ((Trivialization.continuousLinearEquivAt R e b ⋯).symm.trans (Trivializat …
  -/
  ext v
  /-
    case h.h
    R : Type u_1
    B : Type u_2
    F : Type u_3
    E : B → Type u_4
    inst✝¹⁰ : NontriviallyNormedField R
    inst✝⁹ : (x : B) → AddCommMonoid (E x)
    inst✝⁸ : (x : B) → Module R (E x)
    inst✝⁷ : NormedAddCommGroup F
    inst✝⁶ : NormedSpace R F
    inst✝⁵ : TopologicalSpace B
    inst✝⁴ : TopologicalSpace (Bundle.TotalSpace F E)
    inst✝³ : (x : B) → TopologicalSpace (E x)
    inst✝² : FiberBundle F E
    e e' : Trivialization F Bundle.TotalSpace.proj
    inst✝¹ : Trivialization.IsLinear R e
    inst✝ : Trivialization.IsLinear R e'
    b : B
    hb : Membership.mem (Inter.inter e.baseSet e'.baseSet) b
    v : F
    ⊢ Eq (((Trivialization.continuousLinearEquivAt R e b ⋯).symm.trans (Trivializa …
  -/
  rw [coordChangeL_apply e e' hb]
  /-
    case h.h
    R : Type u_1
    B : Type u_2
    F : Type u_3
    E : B → Type u_4
    inst✝¹⁰ : NontriviallyNormedField R
    inst✝⁹ : (x : B) → AddCommMonoid (E x)
    inst✝⁸ : (x : B) → Module R (E x)
    inst✝⁷ : NormedAddCommGroup F
    inst✝⁶ : NormedSpace R F
    inst✝⁵ : TopologicalSpace B
    inst✝⁴ : TopologicalSpace (Bundle.TotalSpace F E)
    inst✝³ : (x : B) → TopologicalSpace (E x)
    inst✝² : FiberBundle F E
    e e' : Trivialization F Bundle.TotalSpace.proj
    inst✝¹ : Trivialization.IsLinear R e
    inst✝ : Trivialization.IsLinear R e'
    b : B
    hb : Membership.mem (Inter.inter e.baseSet e'.baseSet) b
    v : F
    ⊢ Eq (((Trivialization.continuousLinearEquivAt R e b ⋯).symm.trans (Trivializa …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- Analogous construction of `FiberBundleCore` for vector bundles. This
construction gives a way to construct vector bundles from a structure registering how
trivialization changes act on fibers. -/
structure VectorBundleCore (ι : Type*) where
  baseSet : ι → Set B
  isOpen_baseSet : ∀ i, IsOpen (baseSet i)
  indexAt : B → ι
  mem_baseSet_at : ∀ x, x ∈ baseSet (indexAt x)
  coordChange : ι → ι → B → F →L[R] F
  coordChange_self : ∀ i, ∀ x ∈ baseSet i, ∀ v, coordChange i i x v = v
  continuousOn_coordChange : ∀ i j, ContinuousOn (coordChange i j) (baseSet i ∩ baseSet j)
  coordChange_comp : ∀ i j k, ∀ x ∈ baseSet i ∩ baseSet j ∩ baseSet k, ∀ v,
    (coordChange j k x) (coordChange i j x v) = coordChange i k x v


/-- The trivial vector bundle core, in which all the changes of coordinates are the
identity. -/
def trivialVectorBundleCore (ι : Type*) [Inhabited ι] : VectorBundleCore R B F ι where
  baseSet _ := univ
  isOpen_baseSet _ := isOpen_univ
  indexAt := default
  mem_baseSet_at x := mem_univ x
  coordChange _ _ _ := ContinuousLinearMap.id R F
  coordChange_self _ _ _ _ := rfl
  coordChange_comp _ _ _ _ _ _ := rfl
  continuousOn_coordChange _ _ := continuousOn_const


instance (ι : Type*) [Inhabited ι] : Inhabited (VectorBundleCore R B F ι) :=
  ⟨trivialVectorBundleCore R B F ι⟩


/-- Natural identification to a `FiberBundleCore`. -/
@[simps (config := mfld_cfg)]
def toFiberBundleCore : FiberBundleCore ι B F :=
  { Z with
    coordChange := fun i j b => Z.coordChange i j b
    continuousOn_coordChange := fun i j =>
      isBoundedBilinearMap_apply.continuous.comp_continuousOn
        ((Z.continuousOn_coordChange i j).prod_map continuousOn_id) }

-- Porting note (https://github.com/leanprover-community/mathlib4/issues/11215): TODO: restore coercion
-- instance toFiberBundleCoreCoe : Coe (VectorBundleCore R B F ι) (FiberBundleCore ι B F) :=
--   ⟨toFiberBundleCore⟩


theorem coordChange_linear_comp (i j k : ι) :
    ∀ x ∈ Z.baseSet i ∩ Z.baseSet j ∩ Z.baseSet k,
      (Z.coordChange j k x).comp (Z.coordChange i j x) = Z.coordChange i k x :=
  fun x hx => by
  /-
    R : Type u_1
    B : Type u_2
    F : Type u_3
    inst✝³ : NontriviallyNormedField R
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace R F
    inst✝ : TopologicalSpace B
    ι : Type u_5
    Z : VectorBundleCore R B F ι
    i j k : ι
    x : B
    hx : Membership.mem (Inter.inter (Inter.inter (Z.baseSet i) (Z.baseSet j)) (Z. …
    ⊢ Eq ((Z.coordChange j k x).comp (Z.coordChange i j x)) (Z.coordChange i k x)
  -/
  ext v
  /-
    case h
    R : Type u_1
    B : Type u_2
    F : Type u_3
    inst✝³ : NontriviallyNormedField R
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace R F
    inst✝ : TopologicalSpace B
    ι : Type u_5
    Z : VectorBundleCore R B F ι
    i j k : ι
    x : B
    hx : Membership.mem (Inter.inter (Inter.inter (Z.baseSet i) (Z.baseSet j)) (Z. …
    v : F
    ⊢ Eq (((Z.coordChange j k x).comp (Z.coordChange i j x)) v) ((Z.coordChange i  …
  -/
  exact Z.coordChange_comp i j k x hx v
  /-
    🎉 no goals
  -/


/-- The index set of a vector bundle core, as a convenience function for dot notation -/
@[nolint unusedArguments] -- Porting note (https://github.com/leanprover-community/mathlib4/issues/5171): was `nolint has_nonempty_instance`
def Index := ι


/-- The base space of a vector bundle core, as a convenience function for dot notation -/
@[nolint unusedArguments, reducible]
def Base := B


/-- The fiber of a vector bundle core, as a convenience function for dot notation and
typeclass inference -/
@[nolint unusedArguments] -- Porting note (https://github.com/leanprover-community/mathlib4/issues/5171): was `nolint has_nonempty_instance`
def Fiber : B → Type _ :=
  Z.toFiberBundleCore.Fiber


instance topologicalSpaceFiber (x : B) : TopologicalSpace (Z.Fiber x) :=
  Z.toFiberBundleCore.topologicalSpaceFiber x

-- Porting note: fixed: used to assume both `[NormedAddCommGroup F]` and `[AddCommGrp F]`

instance addCommGroupFiber (x : B) : AddCommGroup (Z.Fiber x) :=
  inferInstanceAs (AddCommGroup F)


instance moduleFiber (x : B) : Module R (Z.Fiber x) :=
  inferInstanceAs (Module R F)


/-- The projection from the total space of a fiber bundle core, on its base. -/
@[reducible, simp, mfld_simps]
protected def proj : TotalSpace F Z.Fiber → B :=
  TotalSpace.proj


/-- The total space of the vector bundle, as a convenience function for dot notation.
It is by definition equal to `Bundle.TotalSpace F Z.Fiber`. -/
@[nolint unusedArguments, reducible]
protected def TotalSpace :=
  Bundle.TotalSpace F Z.Fiber


/-- Local homeomorphism version of the trivialization change. -/
def trivChange (i j : ι) : PartialHomeomorph (B × F) (B × F) :=
  Z.toFiberBundleCore.trivChange i j


@[simp, mfld_simps]
theorem mem_trivChange_source (i j : ι) (p : B × F) :
    p ∈ (Z.trivChange i j).source ↔ p.1 ∈ Z.baseSet i ∩ Z.baseSet j :=
  Z.toFiberBundleCore.mem_trivChange_source i j p


/-- Topological structure on the total space of a vector bundle created from core, designed so
that all the local trivialization are continuous. -/
instance toTopologicalSpace : TopologicalSpace Z.TotalSpace :=
  Z.toFiberBundleCore.toTopologicalSpace


@[simp, mfld_simps]
theorem coe_coordChange (i j : ι) : Z.toFiberBundleCore.coordChange i j b = Z.coordChange i j b :=
  rfl


/-- One of the standard local trivializations of a vector bundle constructed from core, taken by
considering this in particular as a fiber bundle constructed from core. -/
def localTriv (i : ι) : Trivialization F (π F Z.Fiber) :=
  Z.toFiberBundleCore.localTriv i

-- Porting note: moved from below to fix the next instance

@[simp, mfld_simps]
theorem localTriv_apply {i : ι} (p : Z.TotalSpace) :
    (Z.localTriv i) p = ⟨p.1, Z.coordChange (Z.indexAt p.1) i p.1 p.2⟩ :=
  rfl


/-- The standard local trivializations of a vector bundle constructed from core are linear. -/
instance localTriv.isLinear (i : ι) : (Z.localTriv i).IsLinear R where
  linear x _ :=
                               /-
                                 R : Type u_1
                                 B : Type u_2
                                 F : Type u_3
                                 E : B → Type u_4
                                 inst✝⁸ : NontriviallyNormedField R
                                 inst✝⁷ : (x : B) → AddCommMonoid (E x)
                                 inst✝⁶ : (x : B) → Module R (E x)
                                 inst✝⁵ : NormedAddCommGroup F
                                 inst✝⁴ : NormedSpace R F
                                 inst✝³ : TopologicalSpace B
                                 inst✝² : TopologicalSpace (Bundle.TotalSpace F E)
                                 inst✝¹ : (x : B) → TopologicalSpace (E x)
                                 inst✝ : FiberBundle F E
                                 ι : Type u_5
                                 Z : VectorBundleCore R B F ι
                                 b : B
                                 a : F
                                 i : ι
                                 x : B
                                 x✝² : Membership.mem (Z.localTriv i).baseSet x
                                 x✝¹ x✝ : Z.Fiber x
                                 ⊢ Eq (↑(Z.localTriv i) { proj := x, snd := HAdd.hAdd x✝¹ x✝ }).2 (HAdd.hAdd (↑ …
                               -/
    { map_add := fun _ _ => by simp only [map_add, localTriv_apply, mfld_simps]
                               /-
                                 🎉 no goals
                               -/
                                /-
                                  R : Type u_1
                                  B : Type u_2
                                  F : Type u_3
                                  E : B → Type u_4
                                  inst✝⁸ : NontriviallyNormedField R
                                  inst✝⁷ : (x : B) → AddCommMonoid (E x)
                                  inst✝⁶ : (x : B) → Module R (E x)
                                  inst✝⁵ : NormedAddCommGroup F
                                  inst✝⁴ : NormedSpace R F
                                  inst✝³ : TopologicalSpace B
                                  inst✝² : TopologicalSpace (Bundle.TotalSpace F E)
                                  inst✝¹ : (x : B) → TopologicalSpace (E x)
                                  inst✝ : FiberBundle F E
                                  ι : Type u_5
                                  Z : VectorBundleCore R B F ι
                                  b : B
                                  a : F
                                  i : ι
                                  x : B
                                  x✝² : Membership.mem (Z.localTriv i).baseSet x
                                  x✝¹ : R
                                  x✝ : Z.Fiber x
                                  ⊢ Eq (↑(Z.localTriv i) { proj := x, snd := HSMul.hSMul x✝¹ x✝ }).2 (HSMul.hSMu …
                                -/
      map_smul := fun _ _ => by simp only [map_smul, localTriv_apply, mfld_simps] }
                                /-
                                  🎉 no goals
                                -/


@[simp, mfld_simps]
theorem mem_localTriv_source (p : Z.TotalSpace) : p ∈ (Z.localTriv i).source ↔ p.1 ∈ Z.baseSet i :=
  Iff.rfl


@[simp, mfld_simps]
theorem baseSet_at : Z.baseSet i = (Z.localTriv i).baseSet :=
  rfl


@[simp, mfld_simps]
theorem mem_localTriv_target (p : B × F) :
    p ∈ (Z.localTriv i).target ↔ p.1 ∈ (Z.localTriv i).baseSet :=
  Z.toFiberBundleCore.mem_localTriv_target i p


@[simp, mfld_simps]
theorem localTriv_symm_fst (p : B × F) :
    (Z.localTriv i).toPartialHomeomorph.symm p = ⟨p.1, Z.coordChange i (Z.indexAt p.1) p.1 p.2⟩ :=
  rfl


@[simp, mfld_simps]
theorem localTriv_symm_apply {b : B} (hb : b ∈ Z.baseSet i) (v : F) :
    (Z.localTriv i).symm b v = Z.coordChange i (Z.indexAt b) b v := by
  /-
    R : Type u_1
    B : Type u_2
    F : Type u_3
    inst✝³ : NontriviallyNormedField R
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace R F
    inst✝ : TopologicalSpace B
    ι : Type u_5
    Z : VectorBundleCore R B F ι
    i : ι
    b : B
    hb : Membership.mem (Z.baseSet i) b
    v : F
    ⊢ Eq ((Z.localTriv i).symm b v) ((Z.coordChange i (Z.indexAt b) b) v)
  -/
  apply (Z.localTriv i).symm_apply hb v
  /-
    🎉 no goals
  -/


@[simp, mfld_simps]
theorem localTriv_coordChange_eq {b : B} (hb : b ∈ Z.baseSet i ∩ Z.baseSet j) (v : F) :
    (Z.localTriv i).coordChangeL R (Z.localTriv j) b v = Z.coordChange i j b v := by
  /-
    R : Type u_1
    B : Type u_2
    F : Type u_3
    inst✝³ : NontriviallyNormedField R
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace R F
    inst✝ : TopologicalSpace B
    ι : Type u_5
    Z : VectorBundleCore R B F ι
    i j : ι
    b : B
    hb : Membership.mem (Inter.inter (Z.baseSet i) (Z.baseSet j)) b
    v : F
    ⊢ Eq ((Trivialization.coordChangeL R (Z.localTriv i) (Z.localTriv j) b) v) ((Z …
  -/
  rw [Trivialization.coordChangeL_apply', localTriv_symm_fst, localTriv_apply, coordChange_comp]
  /-
    case a
    R : Type u_1
    B : Type u_2
    F : Type u_3
    inst✝³ : NontriviallyNormedField R
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace R F
    inst✝ : TopologicalSpace B
    ι : Type u_5
    Z : VectorBundleCore R B F ι
    i j : ι
    b : B
    hb : Membership.mem (Inter.inter (Z.baseSet i) (Z.baseSet j)) b
    v : F
    ⊢ Membership.mem (Inter.inter (Inter.inter (Z.baseSet i) (Z.baseSet (Z.indexAt …
  -/
  exacts [⟨⟨hb.1, Z.mem_baseSet_at b⟩, hb.2⟩, hb]
  /-
    🎉 no goals
  -/


/-- Preferred local trivialization of a vector bundle constructed from core, at a given point, as
a bundle trivialization -/
def localTrivAt (b : B) : Trivialization F (π F Z.Fiber) :=
  Z.localTriv (Z.indexAt b)


@[simp, mfld_simps]
theorem localTrivAt_def : Z.localTriv (Z.indexAt b) = Z.localTrivAt b :=
  rfl


@[simp, mfld_simps]
theorem mem_source_at : (⟨b, a⟩ : Z.TotalSpace) ∈ (Z.localTrivAt b).source := by
  /-
    R : Type u_1
    B : Type u_2
    F : Type u_3
    inst✝³ : NontriviallyNormedField R
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace R F
    inst✝ : TopologicalSpace B
    ι : Type u_5
    Z : VectorBundleCore R B F ι
    b : B
    a : F
    ⊢ Membership.mem (Z.localTrivAt b).source { proj := b, snd := a }
  -/
  rw [localTrivAt, mem_localTriv_source]
  /-
    R : Type u_1
    B : Type u_2
    F : Type u_3
    inst✝³ : NontriviallyNormedField R
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace R F
    inst✝ : TopologicalSpace B
    ι : Type u_5
    Z : VectorBundleCore R B F ι
    b : B
    a : F
    ⊢ Membership.mem (Z.baseSet (Z.indexAt b)) { proj := b, snd := a }.proj
  -/
  exact Z.mem_baseSet_at b
  /-
    🎉 no goals
  -/


@[simp, mfld_simps]
theorem localTrivAt_apply (p : Z.TotalSpace) : Z.localTrivAt p.1 p = ⟨p.1, p.2⟩ :=
  Z.toFiberBundleCore.localTrivAt_apply p


@[simp, mfld_simps]
theorem localTrivAt_apply_mk (b : B) (a : F) : Z.localTrivAt b ⟨b, a⟩ = ⟨b, a⟩ :=
  Z.localTrivAt_apply _


@[simp, mfld_simps]
theorem mem_localTrivAt_baseSet : b ∈ (Z.localTrivAt b).baseSet :=
  Z.toFiberBundleCore.mem_localTrivAt_baseSet b


instance fiberBundle : FiberBundle F Z.Fiber :=
  Z.toFiberBundleCore.fiberBundle


instance vectorBundle : VectorBundle R F Z.Fiber where
  trivialization_linear' := by
    /-
      R : Type u_1
      B : Type u_2
      F : Type u_3
      E : B → Type u_4
      inst✝⁸ : NontriviallyNormedField R
      inst✝⁷ : (x : B) → AddCommMonoid (E x)
      inst✝⁶ : (x : B) → Module R (E x)
      inst✝⁵ : NormedAddCommGroup F
      inst✝⁴ : NormedSpace R F
      inst✝³ : TopologicalSpace B
      inst✝² : TopologicalSpace (Bundle.TotalSpace F E)
      inst✝¹ : (x : B) → TopologicalSpace (E x)
      inst✝ : FiberBundle F E
      ι : Type u_5
      Z : VectorBundleCore R B F ι
      b : B
      a : F
      i j : ι
      ⊢ ∀ (e : Trivialization F Bundle.TotalSpace.proj) [inst : MemTrivializationAtl …
    -/
    rintro _ ⟨i, rfl⟩
    /-
      case mk.intro
      R : Type u_1
      B : Type u_2
      F : Type u_3
      E : B → Type u_4
      inst✝⁸ : NontriviallyNormedField R
      inst✝⁷ : (x : B) → AddCommMonoid (E x)
      inst✝⁶ : (x : B) → Module R (E x)
      inst✝⁵ : NormedAddCommGroup F
      inst✝⁴ : NormedSpace R F
      inst✝³ : TopologicalSpace B
      inst✝² : TopologicalSpace (Bundle.TotalSpace F E)
      inst✝¹ : (x : B) → TopologicalSpace (E x)
      inst✝ : FiberBundle F E
      ι : Type u_5
      Z : VectorBundleCore R B F ι
      b : B
      a : F
      i✝ j i : ι
      ⊢ Trivialization.IsLinear R (Z.toFiberBundleCore.localTriv i)
    -/
    apply localTriv.isLinear
    /-
      🎉 no goals
    -/
  continuousOn_coordChange' := by
    /-
      R : Type u_1
      B : Type u_2
      F : Type u_3
      E : B → Type u_4
      inst✝⁸ : NontriviallyNormedField R
      inst✝⁷ : (x : B) → AddCommMonoid (E x)
      inst✝⁶ : (x : B) → Module R (E x)
      inst✝⁵ : NormedAddCommGroup F
      inst✝⁴ : NormedSpace R F
      inst✝³ : TopologicalSpace B
      inst✝² : TopologicalSpace (Bundle.TotalSpace F E)
      inst✝¹ : (x : B) → TopologicalSpace (E x)
      inst✝ : FiberBundle F E
      ι : Type u_5
      Z : VectorBundleCore R B F ι
      b : B
      a : F
      i j : ι
      ⊢ ∀ (e e' : Trivialization F Bundle.TotalSpace.proj) [inst : MemTrivialization …
    -/
    rintro _ _ ⟨i, rfl⟩ ⟨i', rfl⟩
    /-
      case mk.intro.mk.intro
      R : Type u_1
      B : Type u_2
      F : Type u_3
      E : B → Type u_4
      inst✝⁸ : NontriviallyNormedField R
      inst✝⁷ : (x : B) → AddCommMonoid (E x)
      inst✝⁶ : (x : B) → Module R (E x)
      inst✝⁵ : NormedAddCommGroup F
      inst✝⁴ : NormedSpace R F
      inst✝³ : TopologicalSpace B
      inst✝² : TopologicalSpace (Bundle.TotalSpace F E)
      inst✝¹ : (x : B) → TopologicalSpace (E x)
      inst✝ : FiberBundle F E
      ι : Type u_5
      Z : VectorBundleCore R B F ι
      b : B
      a : F
      i✝ j i i' : ι
      ⊢ ContinuousOn (fun b => ↑(Trivialization.coordChangeL R (Z.toFiberBundleCore. …
    -/
    refine (Z.continuousOn_coordChange i i').congr fun b hb => ?_
    /-
      case mk.intro.mk.intro
      R : Type u_1
      B : Type u_2
      F : Type u_3
      E : B → Type u_4
      inst✝⁸ : NontriviallyNormedField R
      inst✝⁷ : (x : B) → AddCommMonoid (E x)
      inst✝⁶ : (x : B) → Module R (E x)
      inst✝⁵ : NormedAddCommGroup F
      inst✝⁴ : NormedSpace R F
      inst✝³ : TopologicalSpace B
      inst✝² : TopologicalSpace (Bundle.TotalSpace F E)
      inst✝¹ : (x : B) → TopologicalSpace (E x)
      inst✝ : FiberBundle F E
      ι : Type u_5
      Z : VectorBundleCore R B F ι
      b✝ : B
      a : F
      i✝ j i i' : ι
      b : B
      hb : Membership.mem (Inter.inter (Z.baseSet i) (Z.baseSet i')) b
      ⊢ Eq (↑(Trivialization.coordChangeL R (Z.toFiberBundleCore.localTriv i) (Z.toF …
    -/
    ext v
    /-
      case mk.intro.mk.intro.h
      R : Type u_1
      B : Type u_2
      F : Type u_3
      E : B → Type u_4
      inst✝⁸ : NontriviallyNormedField R
      inst✝⁷ : (x : B) → AddCommMonoid (E x)
      inst✝⁶ : (x : B) → Module R (E x)
      inst✝⁵ : NormedAddCommGroup F
      inst✝⁴ : NormedSpace R F
      inst✝³ : TopologicalSpace B
      inst✝² : TopologicalSpace (Bundle.TotalSpace F E)
      inst✝¹ : (x : B) → TopologicalSpace (E x)
      inst✝ : FiberBundle F E
      ι : Type u_5
      Z : VectorBundleCore R B F ι
      b✝ : B
      a : F
      i✝ j i i' : ι
      b : B
      hb : Membership.mem (Inter.inter (Z.baseSet i) (Z.baseSet i')) b
      v : F
      ⊢ Eq (↑(Trivialization.coordChangeL R (Z.toFiberBundleCore.localTriv i) (Z.toF …
    -/
    exact Z.localTriv_coordChange_eq i i' hb v
    /-
      🎉 no goals
    -/


/-- The projection on the base of a vector bundle created from core is continuous -/
@[continuity]
theorem continuous_proj : Continuous Z.proj :=
  Z.toFiberBundleCore.continuous_proj


/-- The projection on the base of a vector bundle created from core is an open map -/
theorem isOpenMap_proj : IsOpenMap Z.proj :=
  Z.toFiberBundleCore.isOpenMap_proj


@[simp, mfld_simps]
theorem localTriv_continuousLinearMapAt {b : B} (hb : b ∈ Z.baseSet i) :
    (Z.localTriv i).continuousLinearMapAt R b = Z.coordChange (Z.indexAt b) i b := by
  /-
    R : Type u_1
    B : Type u_2
    F : Type u_3
    inst✝³ : NontriviallyNormedField R
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace R F
    inst✝ : TopologicalSpace B
    ι : Type u_5
    Z : VectorBundleCore R B F ι
    i : ι
    b : B
    hb : Membership.mem (Z.baseSet i) b
    ⊢ Eq (Trivialization.continuousLinearMapAt R (Z.localTriv i) b) (Z.coordChange …
  -/
  ext1 v
  /-
    case h
    R : Type u_1
    B : Type u_2
    F : Type u_3
    inst✝³ : NontriviallyNormedField R
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace R F
    inst✝ : TopologicalSpace B
    ι : Type u_5
    Z : VectorBundleCore R B F ι
    i : ι
    b : B
    hb : Membership.mem (Z.baseSet i) b
    v : Z.Fiber b
    ⊢ Eq ((Trivialization.continuousLinearMapAt R (Z.localTriv i) b) v) ((Z.coordC …
  -/
  rw [(Z.localTriv i).continuousLinearMapAt_apply R, (Z.localTriv i).coe_linearMapAt_of_mem]
  /-
    case h
    R : Type u_1
    B : Type u_2
    F : Type u_3
    inst✝³ : NontriviallyNormedField R
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace R F
    inst✝ : TopologicalSpace B
    ι : Type u_5
    Z : VectorBundleCore R B F ι
    i : ι
    b : B
    hb : Membership.mem (Z.baseSet i) b
    v : Z.Fiber b
    ⊢ Eq ((fun y => (↑(Z.localTriv i) { proj := b, snd := y }).2) v) ((Z.coordChan …
  -/
  exacts [rfl, hb]
  /-
    🎉 no goals
  -/


@[simp, mfld_simps]
theorem trivializationAt_continuousLinearMapAt {b₀ b : B}
    (hb : b ∈ (trivializationAt F Z.Fiber b₀).baseSet) :
    (trivializationAt F Z.Fiber b₀).continuousLinearMapAt R b =
      Z.coordChange (Z.indexAt b) (Z.indexAt b₀) b :=
  Z.localTriv_continuousLinearMapAt hb


@[simp, mfld_simps]
theorem localTriv_symmL {b : B} (hb : b ∈ Z.baseSet i) :
    (Z.localTriv i).symmL R b = Z.coordChange i (Z.indexAt b) b := by
  /-
    R : Type u_1
    B : Type u_2
    F : Type u_3
    inst✝³ : NontriviallyNormedField R
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace R F
    inst✝ : TopologicalSpace B
    ι : Type u_5
    Z : VectorBundleCore R B F ι
    i : ι
    b : B
    hb : Membership.mem (Z.baseSet i) b
    ⊢ Eq (Trivialization.symmL R (Z.localTriv i) b) (Z.coordChange i (Z.indexAt b) …
  -/
  ext1 v
  /-
    case h
    R : Type u_1
    B : Type u_2
    F : Type u_3
    inst✝³ : NontriviallyNormedField R
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace R F
    inst✝ : TopologicalSpace B
    ι : Type u_5
    Z : VectorBundleCore R B F ι
    i : ι
    b : B
    hb : Membership.mem (Z.baseSet i) b
    v : F
    ⊢ Eq ((Trivialization.symmL R (Z.localTriv i) b) v) ((Z.coordChange i (Z.index …
  -/
  rw [(Z.localTriv i).symmL_apply R, (Z.localTriv i).symm_apply]
  /-
    case h
    R : Type u_1
    B : Type u_2
    F : Type u_3
    inst✝³ : NontriviallyNormedField R
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace R F
    inst✝ : TopologicalSpace B
    ι : Type u_5
    Z : VectorBundleCore R B F ι
    i : ι
    b : B
    hb : Membership.mem (Z.baseSet i) b
    v : F
    ⊢ Eq (cast ⋯ (↑(Z.localTriv i).symm { fst := b, snd := v }).snd) ((Z.coordChan …
  -/
  exacts [rfl, hb]
  /-
    🎉 no goals
  -/


@[simp, mfld_simps]
theorem trivializationAt_symmL {b₀ b : B} (hb : b ∈ (trivializationAt F Z.Fiber b₀).baseSet) :
    (trivializationAt F Z.Fiber b₀).symmL R b = Z.coordChange (Z.indexAt b₀) (Z.indexAt b) b :=
  Z.localTriv_symmL hb


@[simp, mfld_simps]
theorem trivializationAt_coordChange_eq {b₀ b₁ b : B}
    (hb : b ∈ (trivializationAt F Z.Fiber b₀).baseSet ∩ (trivializationAt F Z.Fiber b₁).baseSet)
    (v : F) :
    (trivializationAt F Z.Fiber b₀).coordChangeL R (trivializationAt F Z.Fiber b₁) b v =
      Z.coordChange (Z.indexAt b₀) (Z.indexAt b₁) b v :=
  Z.localTriv_coordChange_eq _ _ hb v


/-- This structure permits to define a vector bundle when trivializations are given as local
equivalences but there is not yet a topology on the total space or the fibers.
The total space is hence given a topology in such a way that there is a fiber bundle structure for
which the partial equivalences are also partial homeomorphisms and hence vector bundle
trivializations. The topology on the fibers is induced from the one on the total space.

The field `exists_coordChange` is stated as an existential statement (instead of 3 separate
fields), since it depends on propositional information (namely `e e' ∈ pretrivializationAtlas`).
This makes it inconvenient to explicitly define a `coordChange` function when constructing a
`VectorPrebundle`. -/
-- Porting note (https://github.com/leanprover-community/mathlib4/issues/5171): was @[nolint has_nonempty_instance]
structure VectorPrebundle where
  pretrivializationAtlas : Set (Pretrivialization F (π F E))
  pretrivialization_linear' : ∀ e, e ∈ pretrivializationAtlas → e.IsLinear R
  pretrivializationAt : B → Pretrivialization F (π F E)
  mem_base_pretrivializationAt : ∀ x : B, x ∈ (pretrivializationAt x).baseSet
  pretrivialization_mem_atlas : ∀ x : B, pretrivializationAt x ∈ pretrivializationAtlas
  exists_coordChange : ∀ᵉ (e ∈ pretrivializationAtlas) (e' ∈ pretrivializationAtlas),
    ∃ f : B → F →L[R] F, ContinuousOn f (e.baseSet ∩ e'.baseSet) ∧
      ∀ᵉ (b ∈ e.baseSet ∩ e'.baseSet) (v : F), f b v = (e' ⟨b, e.symm b v⟩).2
  totalSpaceMk_isInducing : ∀ b : B, IsInducing (pretrivializationAt b ∘ .mk b)


/-- A randomly chosen coordinate change on a `VectorPrebundle`, given by
  the field `exists_coordChange`. -/
def coordChange (a : VectorPrebundle R F E) {e e' : Pretrivialization F (π F E)}
    (he : e ∈ a.pretrivializationAtlas) (he' : e' ∈ a.pretrivializationAtlas) (b : B) : F →L[R] F :=
  Classical.choose (a.exists_coordChange e he e' he') b


theorem continuousOn_coordChange (a : VectorPrebundle R F E) {e e' : Pretrivialization F (π F E)}
    (he : e ∈ a.pretrivializationAtlas) (he' : e' ∈ a.pretrivializationAtlas) :
    ContinuousOn (a.coordChange he he') (e.baseSet ∩ e'.baseSet) :=
  (Classical.choose_spec (a.exists_coordChange e he e' he')).1


theorem coordChange_apply (a : VectorPrebundle R F E) {e e' : Pretrivialization F (π F E)}
    (he : e ∈ a.pretrivializationAtlas) (he' : e' ∈ a.pretrivializationAtlas) {b : B}
    (hb : b ∈ e.baseSet ∩ e'.baseSet) (v : F) :
    a.coordChange he he' b v = (e' ⟨b, e.symm b v⟩).2 :=
  (Classical.choose_spec (a.exists_coordChange e he e' he')).2 b hb v


theorem mk_coordChange (a : VectorPrebundle R F E) {e e' : Pretrivialization F (π F E)}
    (he : e ∈ a.pretrivializationAtlas) (he' : e' ∈ a.pretrivializationAtlas) {b : B}
    (hb : b ∈ e.baseSet ∩ e'.baseSet) (v : F) :
    (b, a.coordChange he he' b v) = e' ⟨b, e.symm b v⟩ := by
  /-
    R : Type u_1
    B : Type u_2
    F : Type u_3
    E : B → Type u_4
    inst✝⁶ : NontriviallyNormedField R
    inst✝⁵ : (x : B) → AddCommMonoid (E x)
    inst✝⁴ : (x : B) → Module R (E x)
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace R F
    inst✝¹ : TopologicalSpace B
    inst✝ : (x : B) → TopologicalSpace (E x)
    a : VectorPrebundle R F E
    e e' : Pretrivialization F Bundle.TotalSpace.proj
    he : Membership.mem a.pretrivializationAtlas e
    he' : Membership.mem a.pretrivializationAtlas e'
    b : B
    hb : Membership.mem (Inter.inter e.baseSet e'.baseSet) b
    v : F
    ⊢ Eq { fst := b, snd := (a.coordChange he he' b) v } (↑e' { proj := b, snd :=  …
  -/
  ext
    /-
      case fst
      R : Type u_1
      B : Type u_2
      F : Type u_3
      E : B → Type u_4
      inst✝⁶ : NontriviallyNormedField R
      inst✝⁵ : (x : B) → AddCommMonoid (E x)
      inst✝⁴ : (x : B) → Module R (E x)
      inst✝³ : NormedAddCommGroup F
      inst✝² : NormedSpace R F
      inst✝¹ : TopologicalSpace B
      inst✝ : (x : B) → TopologicalSpace (E x)
      a : VectorPrebundle R F E
      e e' : Pretrivialization F Bundle.TotalSpace.proj
      he : Membership.mem a.pretrivializationAtlas e
      he' : Membership.mem a.pretrivializationAtlas e'
      b : B
      hb : Membership.mem (Inter.inter e.baseSet e'.baseSet) b
      v : F
      ⊢ Eq { fst := b, snd := (a.coordChange he he' b) v }.1 (↑e' { proj := b, snd : …
    -/
  · rw [e.mk_symm hb.1 v, e'.coe_fst', e.proj_symm_apply' hb.1]
    /-
      case fst
      R : Type u_1
      B : Type u_2
      F : Type u_3
      E : B → Type u_4
      inst✝⁶ : NontriviallyNormedField R
      inst✝⁵ : (x : B) → AddCommMonoid (E x)
      inst✝⁴ : (x : B) → Module R (E x)
      inst✝³ : NormedAddCommGroup F
      inst✝² : NormedSpace R F
      inst✝¹ : TopologicalSpace B
      inst✝ : (x : B) → TopologicalSpace (E x)
      a : VectorPrebundle R F E
      e e' : Pretrivialization F Bundle.TotalSpace.proj
      he : Membership.mem a.pretrivializationAtlas e
      he' : Membership.mem a.pretrivializationAtlas e'
      b : B
      hb : Membership.mem (Inter.inter e.baseSet e'.baseSet) b
      v : F
      ⊢ Membership.mem e'.baseSet (↑e.symm { fst := b, snd := v }).proj
    -/
    rw [e.proj_symm_apply' hb.1]
    /-
      case fst
      R : Type u_1
      B : Type u_2
      F : Type u_3
      E : B → Type u_4
      inst✝⁶ : NontriviallyNormedField R
      inst✝⁵ : (x : B) → AddCommMonoid (E x)
      inst✝⁴ : (x : B) → Module R (E x)
      inst✝³ : NormedAddCommGroup F
      inst✝² : NormedSpace R F
      inst✝¹ : TopologicalSpace B
      inst✝ : (x : B) → TopologicalSpace (E x)
      a : VectorPrebundle R F E
      e e' : Pretrivialization F Bundle.TotalSpace.proj
      he : Membership.mem a.pretrivializationAtlas e
      he' : Membership.mem a.pretrivializationAtlas e'
      b : B
      hb : Membership.mem (Inter.inter e.baseSet e'.baseSet) b
      v : F
      ⊢ Membership.mem e'.baseSet b
    -/
    exact hb.2
    /-
      🎉 no goals
    -/
    /-
      case snd
      R : Type u_1
      B : Type u_2
      F : Type u_3
      E : B → Type u_4
      inst✝⁶ : NontriviallyNormedField R
      inst✝⁵ : (x : B) → AddCommMonoid (E x)
      inst✝⁴ : (x : B) → Module R (E x)
      inst✝³ : NormedAddCommGroup F
      inst✝² : NormedSpace R F
      inst✝¹ : TopologicalSpace B
      inst✝ : (x : B) → TopologicalSpace (E x)
      a : VectorPrebundle R F E
      e e' : Pretrivialization F Bundle.TotalSpace.proj
      he : Membership.mem a.pretrivializationAtlas e
      he' : Membership.mem a.pretrivializationAtlas e'
      b : B
      hb : Membership.mem (Inter.inter e.baseSet e'.baseSet) b
      v : F
      ⊢ Eq { fst := b, snd := (a.coordChange he he' b) v }.2 (↑e' { proj := b, snd : …
    -/
  · exact a.coordChange_apply he he' hb v
    /-
      🎉 no goals
    -/


/-- Natural identification of `VectorPrebundle` as a `FiberPrebundle`. -/
def toFiberPrebundle (a : VectorPrebundle R F E) : FiberPrebundle F E :=
  { a with
    continuous_trivChange := fun e he e' he' ↦ by
      have : ContinuousOn (fun x : B × F ↦ a.coordChange he' he x.1 x.2)
          ((e'.baseSet ∩ e.baseSet) ×ˢ univ) :=
        isBoundedBilinearMap_apply.continuous.comp_continuousOn
          ((a.continuousOn_coordChange he' he).prod_map continuousOn_id)
      /-
        R : Type u_1
        B : Type u_2
        F : Type u_3
        E : B → Type u_4
        inst✝⁶ : NontriviallyNormedField R
        inst✝⁵ : (x : B) → AddCommMonoid (E x)
        inst✝⁴ : (x : B) → Module R (E x)
        inst✝³ : NormedAddCommGroup F
        inst✝² : NormedSpace R F
        inst✝¹ : TopologicalSpace B
        inst✝ : (x : B) → TopologicalSpace (E x)
        a : VectorPrebundle R F E
        e : Pretrivialization F Bundle.TotalSpace.proj
        he : Membership.mem a.pretrivializationAtlas e
        e' : Pretrivialization F Bundle.TotalSpace.proj
        he' : Membership.mem a.pretrivializationAtlas e'
        this : ContinuousOn (fun x => (a.coordChange he' he x.1) x.2) (SProd.sprod (In …
        ⊢ ContinuousOn (Function.comp ↑e ↑e'.symm) (Inter.inter e'.target (Set.preimag …
      -/
      rw [e.target_inter_preimage_symm_source_eq e', inter_comm]
      /-
        R : Type u_1
        B : Type u_2
        F : Type u_3
        E : B → Type u_4
        inst✝⁶ : NontriviallyNormedField R
        inst✝⁵ : (x : B) → AddCommMonoid (E x)
        inst✝⁴ : (x : B) → Module R (E x)
        inst✝³ : NormedAddCommGroup F
        inst✝² : NormedSpace R F
        inst✝¹ : TopologicalSpace B
        inst✝ : (x : B) → TopologicalSpace (E x)
        a : VectorPrebundle R F E
        e : Pretrivialization F Bundle.TotalSpace.proj
        he : Membership.mem a.pretrivializationAtlas e
        e' : Pretrivialization F Bundle.TotalSpace.proj
        he' : Membership.mem a.pretrivializationAtlas e'
        this : ContinuousOn (fun x => (a.coordChange he' he x.1) x.2) (SProd.sprod (In …
        ⊢ ContinuousOn (Function.comp ↑e ↑e'.symm) (SProd.sprod (Inter.inter e'.baseSe …
      -/
      refine (continuousOn_fst.prod this).congr ?_
      /-
        R : Type u_1
        B : Type u_2
        F : Type u_3
        E : B → Type u_4
        inst✝⁶ : NontriviallyNormedField R
        inst✝⁵ : (x : B) → AddCommMonoid (E x)
        inst✝⁴ : (x : B) → Module R (E x)
        inst✝³ : NormedAddCommGroup F
        inst✝² : NormedSpace R F
        inst✝¹ : TopologicalSpace B
        inst✝ : (x : B) → TopologicalSpace (E x)
        a : VectorPrebundle R F E
        e : Pretrivialization F Bundle.TotalSpace.proj
        he : Membership.mem a.pretrivializationAtlas e
        e' : Pretrivialization F Bundle.TotalSpace.proj
        he' : Membership.mem a.pretrivializationAtlas e'
        this : ContinuousOn (fun x => (a.coordChange he' he x.1) x.2) (SProd.sprod (In …
        ⊢ Set.EqOn (Function.comp ↑e ↑e'.symm) (fun x => { fst := x.1, snd := (a.coord …
      -/
      rintro ⟨b, f⟩ ⟨hb, -⟩
      /-
        case mk.intro
        R : Type u_1
        B : Type u_2
        F : Type u_3
        E : B → Type u_4
        inst✝⁶ : NontriviallyNormedField R
        inst✝⁵ : (x : B) → AddCommMonoid (E x)
        inst✝⁴ : (x : B) → Module R (E x)
        inst✝³ : NormedAddCommGroup F
        inst✝² : NormedSpace R F
        inst✝¹ : TopologicalSpace B
        inst✝ : (x : B) → TopologicalSpace (E x)
        a : VectorPrebundle R F E
        e : Pretrivialization F Bundle.TotalSpace.proj
        he : Membership.mem a.pretrivializationAtlas e
        e' : Pretrivialization F Bundle.TotalSpace.proj
        he' : Membership.mem a.pretrivializationAtlas e'
        this : ContinuousOn (fun x => (a.coordChange he' he x.1) x.2) (SProd.sprod (In …
        b : B
        f : F
        hb : Membership.mem (Inter.inter e'.baseSet e.baseSet) { fst := b, snd := f }.1
        ⊢ Eq (Function.comp ↑e ↑e'.symm { fst := b, snd := f }) ((fun x => { fst := x. …
      -/
      dsimp only [Function.comp_def, Prod.map]
      /-
        case mk.intro
        R : Type u_1
        B : Type u_2
        F : Type u_3
        E : B → Type u_4
        inst✝⁶ : NontriviallyNormedField R
        inst✝⁵ : (x : B) → AddCommMonoid (E x)
        inst✝⁴ : (x : B) → Module R (E x)
        inst✝³ : NormedAddCommGroup F
        inst✝² : NormedSpace R F
        inst✝¹ : TopologicalSpace B
        inst✝ : (x : B) → TopologicalSpace (E x)
        a : VectorPrebundle R F E
        e : Pretrivialization F Bundle.TotalSpace.proj
        he : Membership.mem a.pretrivializationAtlas e
        e' : Pretrivialization F Bundle.TotalSpace.proj
        he' : Membership.mem a.pretrivializationAtlas e'
        this : ContinuousOn (fun x => (a.coordChange he' he x.1) x.2) (SProd.sprod (In …
        b : B
        f : F
        hb : Membership.mem (Inter.inter e'.baseSet e.baseSet) { fst := b, snd := f }.1
        ⊢ Eq (↑e (↑e'.symm { fst := b, snd := f })) { fst := b, snd := (a.coordChange  …
      -/
      rw [a.mk_coordChange _ _ hb, e'.mk_symm hb.1] }
      /-
        🎉 no goals
      -/


/-- Topology on the total space that will make the prebundle into a bundle. -/
def totalSpaceTopology (a : VectorPrebundle R F E) : TopologicalSpace (TotalSpace F E) :=
  a.toFiberPrebundle.totalSpaceTopology


/-- Promotion from a `Pretrivialization` in the `pretrivializationAtlas` of a
`VectorPrebundle` to a `Trivialization`. -/
def trivializationOfMemPretrivializationAtlas (a : VectorPrebundle R F E)
    {e : Pretrivialization F (π F E)} (he : e ∈ a.pretrivializationAtlas) :
    @Trivialization B F _ _ _ a.totalSpaceTopology (π F E) :=
  a.toFiberPrebundle.trivializationOfMemPretrivializationAtlas he


theorem linear_trivializationOfMemPretrivializationAtlas (a : VectorPrebundle R F E)
    {e : Pretrivialization F (π F E)} (he : e ∈ a.pretrivializationAtlas) :
    letI := a.totalSpaceTopology
    Trivialization.IsLinear R (trivializationOfMemPretrivializationAtlas a he) :=
  letI := a.totalSpaceTopology
  { linear := (a.pretrivialization_linear' e he).linear }


theorem mem_trivialization_at_source (b : B) (x : E b) :
    ⟨b, x⟩ ∈ (a.pretrivializationAt b).source :=
  a.toFiberPrebundle.mem_pretrivializationAt_source b x


@[simp]
theorem totalSpaceMk_preimage_source (b : B) :
    .mk b ⁻¹' (a.pretrivializationAt b).source = univ :=
  a.toFiberPrebundle.totalSpaceMk_preimage_source b


@[continuity]
theorem continuous_totalSpaceMk (b : B) :
    Continuous[_, a.totalSpaceTopology] (.mk b) :=
  a.toFiberPrebundle.continuous_totalSpaceMk b


/-- Make a `FiberBundle` from a `VectorPrebundle`; auxiliary construction for
`VectorPrebundle.toVectorBundle`. -/
def toFiberBundle : @FiberBundle B F _ _ _ a.totalSpaceTopology _ :=
  a.toFiberPrebundle.toFiberBundle


/-- Make a `VectorBundle` from a `VectorPrebundle`.  Concretely this means
that, given a `VectorPrebundle` structure for a sigma-type `E` -- which consists of a
number of "pretrivializations" identifying parts of `E` with product spaces `U × F` -- one
establishes that for the topology constructed on the sigma-type using
`VectorPrebundle.totalSpaceTopology`, these "pretrivializations" are actually
"trivializations" (i.e., homeomorphisms with respect to the constructed topology). -/
theorem toVectorBundle : @VectorBundle R _ F E _ _ _ _ _ _ a.totalSpaceTopology _ a.toFiberBundle :=
  letI := a.totalSpaceTopology; letI := a.toFiberBundle
  { trivialization_linear' := by
      /-
        R : Type u_1
        B : Type u_2
        F : Type u_3
        E : B → Type u_4
        inst✝⁶ : NontriviallyNormedField R
        inst✝⁵ : (x : B) → AddCommMonoid (E x)
        inst✝⁴ : (x : B) → Module R (E x)
        inst✝³ : NormedAddCommGroup F
        inst✝² : NormedSpace R F
        inst✝¹ : TopologicalSpace B
        inst✝ : (x : B) → TopologicalSpace (E x)
        a : VectorPrebundle R F E
        this✝ : TopologicalSpace (Bundle.TotalSpace F E) := a.totalSpaceTopology
        this : FiberBundle F E := a.toFiberBundle
        ⊢ ∀ (e : Trivialization F Bundle.TotalSpace.proj) [inst : MemTrivializationAtl …
      -/
      rintro _ ⟨e, he, rfl⟩
      /-
        case mk.intro.intro
        R : Type u_1
        B : Type u_2
        F : Type u_3
        E : B → Type u_4
        inst✝⁶ : NontriviallyNormedField R
        inst✝⁵ : (x : B) → AddCommMonoid (E x)
        inst✝⁴ : (x : B) → Module R (E x)
        inst✝³ : NormedAddCommGroup F
        inst✝² : NormedSpace R F
        inst✝¹ : TopologicalSpace B
        inst✝ : (x : B) → TopologicalSpace (E x)
        a : VectorPrebundle R F E
        this✝ : TopologicalSpace (Bundle.TotalSpace F E) := a.totalSpaceTopology
        this : FiberBundle F E := a.toFiberBundle
        e : Pretrivialization F Bundle.TotalSpace.proj
        he : Membership.mem a.toFiberPrebundle.pretrivializationAtlas e
        ⊢ Trivialization.IsLinear R (a.toFiberPrebundle.trivializationOfMemPretriviali …
      -/
      apply linear_trivializationOfMemPretrivializationAtlas
      /-
        🎉 no goals
      -/
    continuousOn_coordChange' := by
      /-
        R : Type u_1
        B : Type u_2
        F : Type u_3
        E : B → Type u_4
        inst✝⁶ : NontriviallyNormedField R
        inst✝⁵ : (x : B) → AddCommMonoid (E x)
        inst✝⁴ : (x : B) → Module R (E x)
        inst✝³ : NormedAddCommGroup F
        inst✝² : NormedSpace R F
        inst✝¹ : TopologicalSpace B
        inst✝ : (x : B) → TopologicalSpace (E x)
        a : VectorPrebundle R F E
        this✝ : TopologicalSpace (Bundle.TotalSpace F E) := a.totalSpaceTopology
        this : FiberBundle F E := a.toFiberBundle
        ⊢ ∀ (e e' : Trivialization F Bundle.TotalSpace.proj) [inst : MemTrivialization …
      -/
      rintro _ _ ⟨e, he, rfl⟩ ⟨e', he', rfl⟩
      /-
        case mk.intro.intro.mk.intro.intro
        R : Type u_1
        B : Type u_2
        F : Type u_3
        E : B → Type u_4
        inst✝⁶ : NontriviallyNormedField R
        inst✝⁵ : (x : B) → AddCommMonoid (E x)
        inst✝⁴ : (x : B) → Module R (E x)
        inst✝³ : NormedAddCommGroup F
        inst✝² : NormedSpace R F
        inst✝¹ : TopologicalSpace B
        inst✝ : (x : B) → TopologicalSpace (E x)
        a : VectorPrebundle R F E
        this✝ : TopologicalSpace (Bundle.TotalSpace F E) := a.totalSpaceTopology
        this : FiberBundle F E := a.toFiberBundle
        e : Pretrivialization F Bundle.TotalSpace.proj
        he : Membership.mem a.toFiberPrebundle.pretrivializationAtlas e
        e' : Pretrivialization F Bundle.TotalSpace.proj
        he' : Membership.mem a.toFiberPrebundle.pretrivializationAtlas e'
        ⊢ ContinuousOn (fun b => ↑(Trivialization.coordChangeL R (a.toFiberPrebundle.t …
      -/
      refine (a.continuousOn_coordChange he he').congr fun b hb ↦ ?_
      /-
        case mk.intro.intro.mk.intro.intro
        R : Type u_1
        B : Type u_2
        F : Type u_3
        E : B → Type u_4
        inst✝⁶ : NontriviallyNormedField R
        inst✝⁵ : (x : B) → AddCommMonoid (E x)
        inst✝⁴ : (x : B) → Module R (E x)
        inst✝³ : NormedAddCommGroup F
        inst✝² : NormedSpace R F
        inst✝¹ : TopologicalSpace B
        inst✝ : (x : B) → TopologicalSpace (E x)
        a : VectorPrebundle R F E
        this✝ : TopologicalSpace (Bundle.TotalSpace F E) := a.totalSpaceTopology
        this : FiberBundle F E := a.toFiberBundle
        e : Pretrivialization F Bundle.TotalSpace.proj
        he : Membership.mem a.toFiberPrebundle.pretrivializationAtlas e
        e' : Pretrivialization F Bundle.TotalSpace.proj
        he' : Membership.mem a.toFiberPrebundle.pretrivializationAtlas e'
        b : B
        hb : Membership.mem (Inter.inter e.baseSet e'.baseSet) b
        ⊢ Eq (↑(Trivialization.coordChangeL R (a.toFiberPrebundle.trivializationOfMemP …
      -/
      ext v
      -- Porting note: help `rw` find instances
      /-
        case mk.intro.intro.mk.intro.intro.h
        R : Type u_1
        B : Type u_2
        F : Type u_3
        E : B → Type u_4
        inst✝⁶ : NontriviallyNormedField R
        inst✝⁵ : (x : B) → AddCommMonoid (E x)
        inst✝⁴ : (x : B) → Module R (E x)
        inst✝³ : NormedAddCommGroup F
        inst✝² : NormedSpace R F
        inst✝¹ : TopologicalSpace B
        inst✝ : (x : B) → TopologicalSpace (E x)
        a : VectorPrebundle R F E
        this✝ : TopologicalSpace (Bundle.TotalSpace F E) := a.totalSpaceTopology
        this : FiberBundle F E := a.toFiberBundle
        e : Pretrivialization F Bundle.TotalSpace.proj
        he : Membership.mem a.toFiberPrebundle.pretrivializationAtlas e
        e' : Pretrivialization F Bundle.TotalSpace.proj
        he' : Membership.mem a.toFiberPrebundle.pretrivializationAtlas e'
        b : B
        hb : Membership.mem (Inter.inter e.baseSet e'.baseSet) b
        v : F
        ⊢ Eq (↑(Trivialization.coordChangeL R (a.toFiberPrebundle.trivializationOfMemP …
      -/
      haveI h₁ := a.linear_trivializationOfMemPretrivializationAtlas he
      /-
        case mk.intro.intro.mk.intro.intro.h
        R : Type u_1
        B : Type u_2
        F : Type u_3
        E : B → Type u_4
        inst✝⁶ : NontriviallyNormedField R
        inst✝⁵ : (x : B) → AddCommMonoid (E x)
        inst✝⁴ : (x : B) → Module R (E x)
        inst✝³ : NormedAddCommGroup F
        inst✝² : NormedSpace R F
        inst✝¹ : TopologicalSpace B
        inst✝ : (x : B) → TopologicalSpace (E x)
        a : VectorPrebundle R F E
        this✝ : TopologicalSpace (Bundle.TotalSpace F E) := a.totalSpaceTopology
        this : FiberBundle F E := a.toFiberBundle
        e : Pretrivialization F Bundle.TotalSpace.proj
        he : Membership.mem a.toFiberPrebundle.pretrivializationAtlas e
        e' : Pretrivialization F Bundle.TotalSpace.proj
        he' : Membership.mem a.toFiberPrebundle.pretrivializationAtlas e'
        b : B
        hb : Membership.mem (Inter.inter e.baseSet e'.baseSet) b
        v : F
        h₁ : Trivialization.IsLinear R (a.trivializationOfMemPretrivializationAtlas he)
        ⊢ Eq (↑(Trivialization.coordChangeL R (a.toFiberPrebundle.trivializationOfMemP …
      -/
      haveI h₂ := a.linear_trivializationOfMemPretrivializationAtlas he'
      /-
        case mk.intro.intro.mk.intro.intro.h
        R : Type u_1
        B : Type u_2
        F : Type u_3
        E : B → Type u_4
        inst✝⁶ : NontriviallyNormedField R
        inst✝⁵ : (x : B) → AddCommMonoid (E x)
        inst✝⁴ : (x : B) → Module R (E x)
        inst✝³ : NormedAddCommGroup F
        inst✝² : NormedSpace R F
        inst✝¹ : TopologicalSpace B
        inst✝ : (x : B) → TopologicalSpace (E x)
        a : VectorPrebundle R F E
        this✝ : TopologicalSpace (Bundle.TotalSpace F E) := a.totalSpaceTopology
        this : FiberBundle F E := a.toFiberBundle
        e : Pretrivialization F Bundle.TotalSpace.proj
        he : Membership.mem a.toFiberPrebundle.pretrivializationAtlas e
        e' : Pretrivialization F Bundle.TotalSpace.proj
        he' : Membership.mem a.toFiberPrebundle.pretrivializationAtlas e'
        b : B
        hb : Membership.mem (Inter.inter e.baseSet e'.baseSet) b
        v : F
        h₁ : Trivialization.IsLinear R (a.trivializationOfMemPretrivializationAtlas he)
        h₂ : Trivialization.IsLinear R (a.trivializationOfMemPretrivializationAtlas he')
        ⊢ Eq (↑(Trivialization.coordChangeL R (a.toFiberPrebundle.trivializationOfMemP …
      -/
      rw [trivializationOfMemPretrivializationAtlas] at h₁ h₂
      rw [a.coordChange_apply he he' hb v, ContinuousLinearEquiv.coe_coe,
        Trivialization.coordChangeL_apply]
      /-
        case mk.intro.intro.mk.intro.intro.h
        R : Type u_1
        B : Type u_2
        F : Type u_3
        E : B → Type u_4
        inst✝⁶ : NontriviallyNormedField R
        inst✝⁵ : (x : B) → AddCommMonoid (E x)
        inst✝⁴ : (x : B) → Module R (E x)
        inst✝³ : NormedAddCommGroup F
        inst✝² : NormedSpace R F
        inst✝¹ : TopologicalSpace B
        inst✝ : (x : B) → TopologicalSpace (E x)
        a : VectorPrebundle R F E
        this✝ : TopologicalSpace (Bundle.TotalSpace F E) := a.totalSpaceTopology
        this : FiberBundle F E := a.toFiberBundle
        e : Pretrivialization F Bundle.TotalSpace.proj
        he : Membership.mem a.toFiberPrebundle.pretrivializationAtlas e
        e' : Pretrivialization F Bundle.TotalSpace.proj
        he' : Membership.mem a.toFiberPrebundle.pretrivializationAtlas e'
        b : B
        hb : Membership.mem (Inter.inter e.baseSet e'.baseSet) b
        v : F
        h₁ : Trivialization.IsLinear R (a.toFiberPrebundle.trivializationOfMemPretrivi …
        h₂ : Trivialization.IsLinear R (a.toFiberPrebundle.trivializationOfMemPretrivi …
        ⊢ Eq (↑(a.toFiberPrebundle.trivializationOfMemPretrivializationAtlas he') { pr …
      -/
      exacts [rfl, hb] }
      /-
        🎉 no goals
      -/


/-- When `ϕ` is a continuous (semi)linear map between the fibers `E x` and `E' y` of two vector
bundles `E` and `E'`, `ContinuousLinearMap.inCoordinates F E F' E' x₀ x y₀ y ϕ` is a coordinate
change of this continuous linear map w.r.t. the chart around `x₀` and the chart around `y₀`.

It is defined by composing `ϕ` with appropriate coordinate changes given by the vector bundles
`E` and `E'`.
We use the operations `Trivialization.continuousLinearMapAt` and `Trivialization.symmL` in the
definition, instead of `Trivialization.continuousLinearEquivAt`, so that
`ContinuousLinearMap.inCoordinates` is defined everywhere (but see
`ContinuousLinearMap.inCoordinates_eq`).

This is the (second component of the) underlying function of a trivialization of the hom-bundle
(see `hom_trivializationAt_apply`). However, note that `ContinuousLinearMap.inCoordinates` is
defined even when `x` and `y` live in different base sets.
Therefore, it is also convenient when working with the hom-bundle between pulled back bundles.
-/
def inCoordinates (x₀ x : B) (y₀ y : B') (ϕ : E x →SL[σ] E' y) : F →SL[σ] F' :=
  ((trivializationAt F' E' y₀).continuousLinearMapAt 𝕜₂ y).comp <|
    ϕ.comp <| (trivializationAt F E x₀).symmL 𝕜₁ x


/-- Rewrite `ContinuousLinearMap.inCoordinates` using continuous linear equivalences. -/
theorem inCoordinates_eq {x₀ x : B} {y₀ y : B'} {ϕ : E x →SL[σ] E' y}
    (hx : x ∈ (trivializationAt F E x₀).baseSet) (hy : y ∈ (trivializationAt F' E' y₀).baseSet) :
    inCoordinates F E F' E' x₀ x y₀ y ϕ =
      ((trivializationAt F' E' y₀).continuousLinearEquivAt 𝕜₂ y hy : E' y →L[𝕜₂] F').comp
        (ϕ.comp <|
          (((trivializationAt F E x₀).continuousLinearEquivAt 𝕜₁ x hx).symm : F →L[𝕜₁] E x)) := by
  /-
    B : Type u_2
    F : Type u_3
    E : B → Type u_4
    inst✝¹⁹ : (x : B) → AddCommMonoid (E x)
    inst✝¹⁸ : NormedAddCommGroup F
    inst✝¹⁷ : TopologicalSpace B
    inst✝¹⁶ : (x : B) → TopologicalSpace (E x)
    𝕜₁ : Type u_5
    𝕜₂ : Type u_6
    inst✝¹⁵ : NontriviallyNormedField 𝕜₁
    inst✝¹⁴ : NontriviallyNormedField 𝕜₂
    σ : RingHom 𝕜₁ 𝕜₂
    B' : Type u_7
    inst✝¹³ : TopologicalSpace B'
    inst✝¹² : NormedSpace 𝕜₁ F
    inst✝¹¹ : (x : B) → Module 𝕜₁ (E x)
    inst✝¹⁰ : TopologicalSpace (Bundle.TotalSpace F E)
    F' : Type u_8
    inst✝⁹ : NormedAddCommGroup F'
    inst✝⁸ : NormedSpace 𝕜₂ F'
    E' : B' → Type u_9
    inst✝⁷ : (x : B') → AddCommMonoid (E' x)
    inst✝⁶ : (x : B') → Module 𝕜₂ (E' x)
    inst✝⁵ : TopologicalSpace (Bundle.TotalSpace F' E')
    inst✝⁴ : FiberBundle F E
    inst✝³ : VectorBundle 𝕜₁ F E
    inst✝² : (x : B') → TopologicalSpace (E' x)
    inst✝¹ : FiberBundle F' E'
    inst✝ : VectorBundle 𝕜₂ F' E'
    x₀ x : B
    y₀ y : B'
    ϕ : ContinuousLinearMap σ (E x) (E' y)
    hx : Membership.mem (FiberBundle.trivializationAt F E x₀).baseSet x
    hy : Membership.mem (FiberBundle.trivializationAt F' E' y₀).baseSet y
    ⊢ Eq (ContinuousLinearMap.inCoordinates F E F' E' x₀ x y₀ y ϕ) ((↑(Trivializat …
  -/
  ext
  simp_rw [inCoordinates, ContinuousLinearMap.coe_comp', ContinuousLinearEquiv.coe_coe,
    Trivialization.coe_continuousLinearEquivAt_eq, Trivialization.symm_continuousLinearEquivAt_eq]


/-- Rewrite `ContinuousLinearMap.inCoordinates` in a `VectorBundleCore`. -/
protected theorem _root_.VectorBundleCore.inCoordinates_eq {ι ι'} (Z : VectorBundleCore 𝕜₁ B F ι)
    (Z' : VectorBundleCore 𝕜₂ B' F' ι') {x₀ x : B} {y₀ y : B'} (ϕ : F →SL[σ] F')
    (hx : x ∈ Z.baseSet (Z.indexAt x₀)) (hy : y ∈ Z'.baseSet (Z'.indexAt y₀)) :
    inCoordinates F Z.Fiber F' Z'.Fiber x₀ x y₀ y ϕ =
      (Z'.coordChange (Z'.indexAt y) (Z'.indexAt y₀) y).comp
        (ϕ.comp <| Z.coordChange (Z.indexAt x₀) (Z.indexAt x) x) := by
  simp_rw [inCoordinates, Z'.trivializationAt_continuousLinearMapAt hy,
    Z.trivializationAt_symmL hx]


