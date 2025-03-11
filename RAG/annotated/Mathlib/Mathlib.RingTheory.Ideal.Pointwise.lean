/-- The action on an ideal corresponding to applying the action to every element.

This is available as an instance in the `Pointwise` locale. -/
protected def pointwiseDistribMulAction : DistribMulAction M (Ideal R) where
  smul a := Ideal.map (MulSemiringAction.toRingHom _ _ a)
  one_smul I :=
    congr_arg (I.map ·) (RingHom.ext <| one_smul M) |>.trans I.map_id
  mul_smul _ _ I :=
    congr_arg (I.map ·) (RingHom.ext <| mul_smul _ _) |>.trans (I.map_map _ _).symm
  smul_zero _ := Ideal.map_bot
  smul_add _ I J := Ideal.map_sup _ I J


/-- The action on an ideal corresponding to applying the action to every element.

This is available as an instance in the `Pointwise` locale. -/
protected def pointwiseMulSemiringAction {R : Type*} [CommRing R] [MulSemiringAction M R] :
    MulSemiringAction M (Ideal R) where
                   /-
                     M : Type u_1
                     R✝ : Type u_2
                     inst✝⁴ : Monoid M
                     inst✝³ : Semiring R✝
                     inst✝² : MulSemiringAction M R✝
                     R : Type u_3
                     inst✝¹ : CommRing R
                     inst✝ : MulSemiringAction M R
                     a : M
                     ⊢ Eq (HSMul.hSMul a 1) 1
                   -/
  smul_one a := by simp only [Ideal.one_eq_top]; exact Ideal.map_top _
                                                 /-
                                                   🎉 no goals
                                                 -/
  smul_mul a I J := Ideal.map_mul (MulSemiringAction.toRingHom _ _ a) I J


theorem pointwise_smul_def {a : M} (S : Ideal R) :
    a • S = S.map (MulSemiringAction.toRingHom _ _ a) :=
  rfl

-- note: unlike with `Subring`, `pointwise_smul_toAddSubgroup` wouldn't be true


theorem smul_mem_pointwise_smul (m : M) (r : R) (S : Ideal R) : r ∈ S → m • r ∈ m • S :=
  fun h => subset_span <| Set.smul_mem_smul_set h


instance : CovariantClass M (Ideal R) HSMul.hSMul LE.le :=
  ⟨fun _ _ => map_mono⟩

-- note: unlike with `Subring`, `mem_smul_pointwise_iff_exists` wouldn't be true


@[simp]
theorem smul_bot (a : M) : a • (⊥ : Ideal R) = ⊥ :=
  map_bot


theorem smul_sup (a : M) (S T : Ideal R) : a • (S ⊔ T) = a • S ⊔ a • T :=
  map_sup _ _ _


theorem smul_closure (a : M) (s : Set R) : a • span s = span (a • s) :=
  Ideal.map_span _ _


instance pointwise_central_scalar [MulSemiringAction Mᵐᵒᵖ R] [IsCentralScalar M R] :
    IsCentralScalar M (Ideal R) :=
  ⟨fun _ S => (congr_arg fun f => S.map f) <| RingHom.ext <| op_smul_eq_smul _⟩


@[simp]
theorem pointwise_smul_toAddSubmonoid (a : M) (S : Ideal R)
    (ha : Function.Surjective fun r : R => a • r) :
    (a • S).toAddSubmonoid = a • S.toAddSubmonoid := by
  /-
    M : Type u_1
    R : Type u_2
    inst✝² : Monoid M
    inst✝¹ : Semiring R
    inst✝ : MulSemiringAction M R
    a : M
    S : Ideal R
    ha : Function.Surjective fun r => HSMul.hSMul a r
    ⊢ Eq (HSMul.hSMul a S).toAddSubmonoid (HSMul.hSMul a S.toAddSubmonoid)
  -/
  ext
  /-
    case h
    M : Type u_1
    R : Type u_2
    inst✝² : Monoid M
    inst✝¹ : Semiring R
    inst✝ : MulSemiringAction M R
    a : M
    S : Ideal R
    ha : Function.Surjective fun r => HSMul.hSMul a r
    x✝ : R
    ⊢ Iff (Membership.mem (HSMul.hSMul a S).toAddSubmonoid x✝) (Membership.mem (HS …
  -/
  exact Ideal.mem_map_iff_of_surjective _ <| by exact ha
  /-
    🎉 no goals
  -/


@[simp]
theorem pointwise_smul_toAddSubGroup {R : Type*} [Ring R] [MulSemiringAction M R]
    (a : M) (S : Ideal R) (ha : Function.Surjective fun r : R => a • r)  :
    (a • S).toAddSubgroup = a • S.toAddSubgroup := by
  /-
    M : Type u_1
    inst✝² : Monoid M
    R : Type u_3
    inst✝¹ : Ring R
    inst✝ : MulSemiringAction M R
    a : M
    S : Ideal R
    ha : Function.Surjective fun r => HSMul.hSMul a r
    ⊢ Eq (Submodule.toAddSubgroup (HSMul.hSMul a S)) (HSMul.hSMul a (Submodule.toA …
  -/
  ext
  /-
    case h
    M : Type u_1
    inst✝² : Monoid M
    R : Type u_3
    inst✝¹ : Ring R
    inst✝ : MulSemiringAction M R
    a : M
    S : Ideal R
    ha : Function.Surjective fun r => HSMul.hSMul a r
    x✝ : R
    ⊢ Iff (Membership.mem (Submodule.toAddSubgroup (HSMul.hSMul a S)) x✝) (Members …
  -/
  exact Ideal.mem_map_iff_of_surjective _ <| by exact ha
  /-
    🎉 no goals
  -/


theorem pointwise_smul_eq_comap {a : M} (S : Ideal R) :
    a • S = S.comap (MulSemiringAction.toRingAut _ _ a).symm := by
  /-
    M : Type u_1
    R : Type u_2
    inst✝² : Group M
    inst✝¹ : Semiring R
    inst✝ : MulSemiringAction M R
    a : M
    S : Ideal R
    ⊢ Eq (HSMul.hSMul a S) (Ideal.comap (RingEquiv.symm ((MulSemiringAction.toRing …
  -/
  ext
  /-
    case h
    M : Type u_1
    R : Type u_2
    inst✝² : Group M
    inst✝¹ : Semiring R
    inst✝ : MulSemiringAction M R
    a : M
    S : Ideal R
    x✝ : R
    ⊢ Iff (Membership.mem (HSMul.hSMul a S) x✝) (Membership.mem (Ideal.comap (Ring …
  -/
  simp [pointwise_smul_def]
  /-
    case h
    M : Type u_1
    R : Type u_2
    inst✝² : Group M
    inst✝¹ : Semiring R
    inst✝ : MulSemiringAction M R
    a : M
    S : Ideal R
    x✝ : R
    ⊢ Iff (Membership.mem (Ideal.map (MulSemiringAction.toRingHom M R a) S) x✝) (M …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem smul_mem_pointwise_smul_iff {a : M} {S : Ideal R} {x : R} : a • x ∈ a • S ↔ x ∈ S :=
               /-
                 M : Type u_1
                 R : Type u_2
                 inst✝² : Group M
                 inst✝¹ : Semiring R
                 inst✝ : MulSemiringAction M R
                 a : M
                 S : Ideal R
                 x : R
                 h : Membership.mem (HSMul.hSMul a S) (HSMul.hSMul a x)
                 ⊢ Membership.mem S x
               -/
  ⟨fun h => by simpa using smul_mem_pointwise_smul a⁻¹ _ _ h, smul_mem_pointwise_smul _ _ _⟩
               /-
                 🎉 no goals
               -/


theorem mem_pointwise_smul_iff_inv_smul_mem {a : M} {S : Ideal R} {x : R} :
    x ∈ a • S ↔ a⁻¹ • x ∈ S :=
               /-
                 M : Type u_1
                 R : Type u_2
                 inst✝² : Group M
                 inst✝¹ : Semiring R
                 inst✝ : MulSemiringAction M R
                 a : M
                 S : Ideal R
                 x : R
                 h : Membership.mem (HSMul.hSMul a S) x
                 ⊢ Membership.mem S (HSMul.hSMul (Inv.inv a) x)
               -/
  ⟨fun h => by simpa using smul_mem_pointwise_smul a⁻¹ _ _ h,
               /-
                 🎉 no goals
               -/
                /-
                  M : Type u_1
                  R : Type u_2
                  inst✝² : Group M
                  inst✝¹ : Semiring R
                  inst✝ : MulSemiringAction M R
                  a : M
                  S : Ideal R
                  x : R
                  h : Membership.mem S (HSMul.hSMul (Inv.inv a) x)
                  ⊢ Membership.mem (HSMul.hSMul a S) x
                -/
    fun h => by simpa using smul_mem_pointwise_smul a _ _ h⟩
                /-
                  🎉 no goals
                -/


theorem mem_inv_pointwise_smul_iff {a : M} {S : Ideal R} {x : R} : x ∈ a⁻¹ • S ↔ a • x ∈ S := by
  /-
    M : Type u_1
    R : Type u_2
    inst✝² : Group M
    inst✝¹ : Semiring R
    inst✝ : MulSemiringAction M R
    a : M
    S : Ideal R
    x : R
    ⊢ Iff (Membership.mem (HSMul.hSMul (Inv.inv a) S) x) (Membership.mem S (HSMul. …
  -/
  rw [mem_pointwise_smul_iff_inv_smul_mem, inv_inv]
  /-
    🎉 no goals
  -/


@[simp]
theorem pointwise_smul_le_pointwise_smul_iff {a : M} {S T : Ideal R} : a • S ≤ a • T ↔ S ≤ T :=
               /-
                 M : Type u_1
                 R : Type u_2
                 inst✝² : Group M
                 inst✝¹ : Semiring R
                 inst✝ : MulSemiringAction M R
                 a : M
                 S T : Ideal R
                 h : LE.le (HSMul.hSMul a S) (HSMul.hSMul a T)
                 ⊢ LE.le S T
               -/
  ⟨fun h => by simpa using smul_mono_right a⁻¹ h, fun h => smul_mono_right a h⟩
               /-
                 🎉 no goals
               -/


theorem pointwise_smul_subset_iff {a : M} {S T : Ideal R} : a • S ≤ T ↔ S ≤ a⁻¹ • T := by
  /-
    M : Type u_1
    R : Type u_2
    inst✝² : Group M
    inst✝¹ : Semiring R
    inst✝ : MulSemiringAction M R
    a : M
    S T : Ideal R
    ⊢ Iff (LE.le (HSMul.hSMul a S) T) (LE.le S (HSMul.hSMul (Inv.inv a) T))
  -/
  rw [← pointwise_smul_le_pointwise_smul_iff (a := a⁻¹), inv_smul_smul]
  /-
    🎉 no goals
  -/


theorem subset_pointwise_smul_iff {a : M} {S T : Ideal R} : S ≤ a • T ↔ a⁻¹ • S ≤ T := by
  /-
    M : Type u_1
    R : Type u_2
    inst✝² : Group M
    inst✝¹ : Semiring R
    inst✝ : MulSemiringAction M R
    a : M
    S T : Ideal R
    ⊢ Iff (LE.le S (HSMul.hSMul a T)) (LE.le (HSMul.hSMul (Inv.inv a) S) T)
  -/
  rw [← pointwise_smul_le_pointwise_smul_iff (a := a⁻¹), inv_smul_smul]
  /-
    🎉 no goals
  -/


instance IsPrime.smul {I : Ideal R} [H : I.IsPrime] (g : M) : (g • I).IsPrime := by
  /-
    M : Type u_1
    R : Type u_2
    inst✝² : Group M
    inst✝¹ : Semiring R
    inst✝ : MulSemiringAction M R
    I : Ideal R
    H : I.IsPrime
    g : M
    ⊢ (HSMul.hSMul g I).IsPrime
  -/
  rw [I.pointwise_smul_eq_comap]
  /-
    M : Type u_1
    R : Type u_2
    inst✝² : Group M
    inst✝¹ : Semiring R
    inst✝ : MulSemiringAction M R
    I : Ideal R
    H : I.IsPrime
    g : M
    ⊢ (Ideal.comap (RingEquiv.symm ((MulSemiringAction.toRingAut M R) g)) I).IsPrime
  -/
  apply H.comap
  /-
    🎉 no goals
  -/


@[simp]
theorem IsPrime.smul_iff {I : Ideal R} (g : M) : (g • I).IsPrime ↔ I.IsPrime :=
  ⟨fun H ↦ inv_smul_smul g I ▸ H.smul g⁻¹, fun H ↦ H.smul g⟩


