/-- Pull a subalgebra back to an opposite subalgebra along `MulOpposite.unop` -/
@[simps! coe toSubsemiring]
protected def op (S : Subalgebra R A) : Subalgebra R Aᵐᵒᵖ where
  toSubsemiring := S.toSubsemiring.op
  algebraMap_mem' := S.algebraMap_mem


attribute [norm_cast] coe_op


@[simp]
theorem mem_op {x : Aᵐᵒᵖ} {S : Subalgebra R A} : x ∈ S.op ↔ x.unop ∈ S := Iff.rfl


/-- Pull an subalgebra subring back to a subalgebra along `MulOpposite.op` -/
@[simps! coe toSubsemiring]
protected def unop (S : Subalgebra R Aᵐᵒᵖ) : Subalgebra R A where
  toSubsemiring := S.toSubsemiring.unop
  algebraMap_mem' := S.algebraMap_mem


attribute [norm_cast] coe_unop


@[simp]
theorem mem_unop {x : A} {S : Subalgebra R Aᵐᵒᵖ} : x ∈ S.unop ↔ MulOpposite.op x ∈ S := Iff.rfl


@[simp]
theorem unop_op (S : Subalgebra R A) : S.op.unop = S := rfl


@[simp]
theorem op_unop (S : Subalgebra R Aᵐᵒᵖ) : S.unop.op = S := rfl


theorem op_le_iff {S₁ : Subalgebra R A} {S₂ : Subalgebra R Aᵐᵒᵖ} : S₁.op ≤ S₂ ↔ S₁ ≤ S₂.unop :=
  MulOpposite.op_surjective.forall


theorem le_op_iff {S₁ : Subalgebra R Aᵐᵒᵖ} {S₂ : Subalgebra R A} : S₁ ≤ S₂.op ↔ S₁.unop ≤ S₂ :=
  MulOpposite.op_surjective.forall


@[simp]
theorem op_le_op_iff {S₁ S₂ : Subalgebra R A} : S₁.op ≤ S₂.op ↔ S₁ ≤ S₂ :=
  MulOpposite.op_surjective.forall


@[simp]
theorem unop_le_unop_iff {S₁ S₂ : Subalgebra R Aᵐᵒᵖ} : S₁.unop ≤ S₂.unop ↔ S₁ ≤ S₂ :=
  MulOpposite.unop_surjective.forall


/-- A subalgebra `S` of `A / R` determines a subring `S.op` of the opposite ring `Aᵐᵒᵖ / R`. -/
@[simps]
def opEquiv : Subalgebra R A ≃o Subalgebra R Aᵐᵒᵖ where
  toFun := Subalgebra.op
  invFun := Subalgebra.unop
  left_inv := unop_op
  right_inv := op_unop
  map_rel_iff' := op_le_op_iff


@[simp]
theorem op_bot : (⊥ : Subalgebra R A).op = ⊥ := opEquiv.map_bot


@[simp]
theorem unop_bot : (⊥ : Subalgebra R Aᵐᵒᵖ).unop = ⊥ := opEquiv.symm.map_bot


@[simp]
theorem op_top : (⊤ : Subalgebra R A).op = ⊤ := opEquiv.map_top


@[simp]
theorem unop_top : (⊤ : Subalgebra R Aᵐᵒᵖ).unop = ⊤ := opEquiv.symm.map_top


theorem op_sup (S₁ S₂ : Subalgebra R A) : (S₁ ⊔ S₂).op = S₁.op ⊔ S₂.op :=
  opEquiv.map_sup _ _


theorem unop_sup (S₁ S₂ : Subalgebra R Aᵐᵒᵖ) : (S₁ ⊔ S₂).unop = S₁.unop ⊔ S₂.unop :=
  opEquiv.symm.map_sup _ _


theorem op_inf (S₁ S₂ : Subalgebra R A) : (S₁ ⊓ S₂).op = S₁.op ⊓ S₂.op := opEquiv.map_inf _ _


theorem unop_inf (S₁ S₂ : Subalgebra R Aᵐᵒᵖ) : (S₁ ⊓ S₂).unop = S₁.unop ⊓ S₂.unop :=
  opEquiv.symm.map_inf _ _


theorem op_sSup (S : Set (Subalgebra R A)) : (sSup S).op = sSup (.unop ⁻¹' S) :=
  opEquiv.map_sSup_eq_sSup_symm_preimage _


theorem unop_sSup (S : Set (Subalgebra R Aᵐᵒᵖ)) : (sSup S).unop = sSup (.op ⁻¹' S) :=
  opEquiv.symm.map_sSup_eq_sSup_symm_preimage _


theorem op_sInf (S : Set (Subalgebra R A)) : (sInf S).op = sInf (.unop ⁻¹' S) :=
  opEquiv.map_sInf_eq_sInf_symm_preimage _


theorem unop_sInf (S : Set (Subalgebra R Aᵐᵒᵖ)) : (sInf S).unop = sInf (.op ⁻¹' S) :=
  opEquiv.symm.map_sInf_eq_sInf_symm_preimage _


theorem op_iSup (S : ι → Subalgebra R A) : (iSup S).op = ⨆ i, (S i).op := opEquiv.map_iSup _


theorem unop_iSup (S : ι → Subalgebra R Aᵐᵒᵖ) : (iSup S).unop = ⨆ i, (S i).unop :=
  opEquiv.symm.map_iSup _


theorem op_iInf (S : ι → Subalgebra R A) : (iInf S).op = ⨅ i, (S i).op := opEquiv.map_iInf _


theorem unop_iInf (S : ι → Subalgebra R Aᵐᵒᵖ) : (iInf S).unop = ⨅ i, (S i).unop :=
  opEquiv.symm.map_iInf _


theorem op_adjoin (s : Set A) :
    (Algebra.adjoin R s).op = Algebra.adjoin R (MulOpposite.unop ⁻¹' s) := by
  /-
    R : Type u_2
    A : Type u_3
    inst✝² : CommSemiring R
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    s : Set A
    ⊢ Eq (Algebra.adjoin R s).op (Algebra.adjoin R (Set.preimage MulOpposite.unop  …
  -/
  apply toSubsemiring_injective
  /-
    case a
    R : Type u_2
    A : Type u_3
    inst✝² : CommSemiring R
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    s : Set A
    ⊢ Eq (Algebra.adjoin R s).op.toSubsemiring (Algebra.adjoin R (Set.preimage Mul …
  -/
  simp_rw [Algebra.adjoin, op_toSubsemiring, Subsemiring.op_closure, Set.preimage_union]
  /-
    case a
    R : Type u_2
    A : Type u_3
    inst✝² : CommSemiring R
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    s : Set A
    ⊢ Eq (Subsemiring.closure (Union.union (Set.preimage MulOpposite.unop (Set.ran …
  -/
  congr with x
  /-
    case a.e_s.e_a.h
    R : Type u_2
    A : Type u_3
    inst✝² : CommSemiring R
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    s : Set A
    x : MulOpposite A
    ⊢ Iff (Membership.mem (Set.preimage MulOpposite.unop (Set.range ⇑(algebraMap R …
  -/
  simp_rw [Set.mem_preimage, Set.mem_range, MulOpposite.algebraMap_apply]
  /-
    case a.e_s.e_a.h
    R : Type u_2
    A : Type u_3
    inst✝² : CommSemiring R
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    s : Set A
    x : MulOpposite A
    ⊢ Iff (Exists fun y => Eq ((algebraMap R A) y) (MulOpposite.unop x)) (Exists f …
  -/
  congr!
  /-
    case a.e_s.e_a.h.a.h.e'_2.h.a
    R : Type u_2
    A : Type u_3
    inst✝² : CommSemiring R
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    s : Set A
    x : MulOpposite A
    x✝ : R
    ⊢ Iff (Eq ((algebraMap R A) x✝) (MulOpposite.unop x)) (Eq (MulOpposite.op ((al …
  -/
  rw [← MulOpposite.op_injective.eq_iff (b := x.unop), MulOpposite.op_unop]
  /-
    🎉 no goals
  -/


theorem unop_adjoin (s : Set Aᵐᵒᵖ) :
    (Algebra.adjoin R s).unop = Algebra.adjoin R (MulOpposite.op ⁻¹' s) := by
  /-
    R : Type u_2
    A : Type u_3
    inst✝² : CommSemiring R
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    s : Set (MulOpposite A)
    ⊢ Eq (Algebra.adjoin R s).unop (Algebra.adjoin R (Set.preimage MulOpposite.op  …
  -/
  apply toSubsemiring_injective
  /-
    case a
    R : Type u_2
    A : Type u_3
    inst✝² : CommSemiring R
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    s : Set (MulOpposite A)
    ⊢ Eq (Algebra.adjoin R s).unop.toSubsemiring (Algebra.adjoin R (Set.preimage M …
  -/
  simp_rw [Algebra.adjoin, unop_toSubsemiring, Subsemiring.unop_closure, Set.preimage_union]
  /-
    case a
    R : Type u_2
    A : Type u_3
    inst✝² : CommSemiring R
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    s : Set (MulOpposite A)
    ⊢ Eq (Subsemiring.closure (Union.union (Set.preimage MulOpposite.op (Set.range …
  -/
  congr with x
  /-
    case a.e_s.e_a.h
    R : Type u_2
    A : Type u_3
    inst✝² : CommSemiring R
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    s : Set (MulOpposite A)
    x : A
    ⊢ Iff (Membership.mem (Set.preimage MulOpposite.op (Set.range ⇑(algebraMap R ( …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- Bijection between a subalgebra `S` and its opposite. -/
@[simps!]
def linearEquivOp (S : Subalgebra R A) : S ≃ₗ[R] S.op where
  __ := S.toSubsemiring.addEquivOp
  map_smul' _ _ := rfl


/-- Bijection between a subalgebra `S` and `MulOpposite` of its opposite. -/
@[simps!]
def algEquivOpMop (S : Subalgebra R A) : S ≃ₐ[R] (S.op)ᵐᵒᵖ where
  __ := S.toSubsemiring.ringEquivOpMop
  commutes' _ := rfl


/-- Bijection between `MulOpposite` of a subalgebra `S` and its opposite. -/
@[simps!]
def mopAlgEquivOp (S : Subalgebra R A) : Sᵐᵒᵖ ≃ₐ[R] S.op where
  __ := S.toSubsemiring.mopRingEquivOp
  commutes' _ := rfl


@[simp]
theorem op_toSubring (S : Subalgebra R A) : S.op.toSubring = S.toSubring.op := rfl


@[simp]
theorem unop_toSubring (S : Subalgebra R Aᵐᵒᵖ) : S.unop.toSubring = S.toSubring.unop := rfl


