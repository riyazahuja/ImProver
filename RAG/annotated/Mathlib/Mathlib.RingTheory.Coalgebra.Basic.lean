/-- Data fields for `Coalgebra`, to allow API to be constructed before proving `Coalgebra.coassoc`.

See `Coalgebra` for documentation. -/
class CoalgebraStruct (R : Type u) (A : Type v)
    [CommSemiring R] [AddCommMonoid A] [Module R A] where
  /-- The comultiplication of the coalgebra -/
  comul : A →ₗ[R] A ⊗[R] A
  /-- The counit of the coalgebra -/
  counit : A →ₗ[R] R


/--
A representation of an element `a` of a coalgebra `A` is a finite sum of pure tensors `∑ xᵢ ⊗ yᵢ`
that is equal to `comul a`.
-/
structure Coalgebra.Repr (R : Type u) {A : Type v}
    [CommSemiring R] [AddCommMonoid A] [Module R A] [CoalgebraStruct R A] (a : A) where
  /-- the indexing type of a representation of `comul a` -/
  {ι : Type*}
  /-- the finite indexing set of a representation of `comul a` -/
  (index : Finset ι)
  /-- the first coordinate of a representation of `comul a` -/
  (left : ι → A)
  /-- the second coordinate of a representation of `comul a` -/
  (right : ι → A)
  /-- `comul a` is equal to a finite sum of some pure tensors -/
  (eq : ∑ i ∈ index, left i ⊗ₜ[R] right i = CoalgebraStruct.comul a)


/-- An arbitrarily chosen representation. -/
def Coalgebra.Repr.arbitrary (R : Type u) {A : Type v}
    [CommSemiring R] [AddCommMonoid A] [Module R A] [CoalgebraStruct R A] (a : A) :
    Coalgebra.Repr R a where
  index := TensorProduct.exists_finset (R := R) (CoalgebraStruct.comul a) |>.choose
  eq := TensorProduct.exists_finset (R := R) (CoalgebraStruct.comul a) |>.choose_spec.symm


@[inherit_doc Coalgebra.Repr.arbitrary]
scoped[Coalgebra] notation "ℛ" => Coalgebra.Repr.arbitrary


/-- A coalgebra over a commutative (semi)ring `R` is an `R`-module equipped with a coassociative
comultiplication `Δ` and a counit `ε` obeying the left and right counitality laws. -/
class Coalgebra (R : Type u) (A : Type v)
    [CommSemiring R] [AddCommMonoid A] [Module R A] extends CoalgebraStruct R A where
  /-- The comultiplication is coassociative -/
  coassoc : TensorProduct.assoc R A A A ∘ₗ comul.rTensor A ∘ₗ comul = comul.lTensor A ∘ₗ comul
  /-- The counit satisfies the left counitality law -/
  rTensor_counit_comp_comul : counit.rTensor A ∘ₗ comul = TensorProduct.mk R _ _ 1
  /-- The counit satisfies the right counitality law -/
  lTensor_counit_comp_comul : counit.lTensor A ∘ₗ comul = (TensorProduct.mk R _ _).flip 1


@[simp]
theorem coassoc_apply (a : A) :
    TensorProduct.assoc R A A A (comul.rTensor A (comul a)) = comul.lTensor A (comul a) :=
  LinearMap.congr_fun coassoc a


@[simp]
theorem coassoc_symm_apply (a : A) :
    (TensorProduct.assoc R A A A).symm (comul.lTensor A (comul a)) = comul.rTensor A (comul a) := by
  /-
    R : Type u
    A : Type v
    inst✝³ : CommSemiring R
    inst✝² : AddCommMonoid A
    inst✝¹ : Module R A
    inst✝ : Coalgebra R A
    a : A
    ⊢ Eq ((TensorProduct.assoc R A A A).symm ((LinearMap.lTensor A CoalgebraStruct …
  -/
  rw [(TensorProduct.assoc R A A A).symm_apply_eq, coassoc_apply a]
  /-
    🎉 no goals
  -/


@[simp]
theorem coassoc_symm :
    (TensorProduct.assoc R A A A).symm ∘ₗ comul.lTensor A ∘ₗ comul =
    comul.rTensor A ∘ₗ (comul (R := R)) :=
  LinearMap.ext coassoc_symm_apply


@[simp]
theorem rTensor_counit_comul (a : A) : counit.rTensor A (comul a) = 1 ⊗ₜ[R] a :=
  LinearMap.congr_fun rTensor_counit_comp_comul a


@[simp]
theorem lTensor_counit_comul (a : A) : counit.lTensor A (comul a) = a ⊗ₜ[R] 1 :=
  LinearMap.congr_fun lTensor_counit_comp_comul a


@[simp]
lemma sum_counit_tmul_eq {a : A} (repr : Coalgebra.Repr R a) :
    ∑ i ∈ repr.index, counit (R := R) (repr.left i) ⊗ₜ (repr.right i) = 1 ⊗ₜ[R] a := by
  /-
    R : Type u
    A : Type v
    inst✝³ : CommSemiring R
    inst✝² : AddCommMonoid A
    inst✝¹ : Module R A
    inst✝ : Coalgebra R A
    a : A
    repr : Coalgebra.Repr R a
    ⊢ Eq (repr.index.sum fun i => TensorProduct.tmul R (CoalgebraStruct.counit (re …
  -/
  simpa [← repr.eq, map_sum] using congr($(rTensor_counit_comp_comul (R := R) (A := A)) a)
  /-
    🎉 no goals
  -/


@[simp]
lemma sum_tmul_counit_eq {a : A} (repr : Coalgebra.Repr R a) :
    ∑ i ∈ repr.index, (repr.left i) ⊗ₜ counit (R := R) (repr.right i) = a ⊗ₜ[R] 1 := by
  /-
    R : Type u
    A : Type v
    inst✝³ : CommSemiring R
    inst✝² : AddCommMonoid A
    inst✝¹ : Module R A
    inst✝ : Coalgebra R A
    a : A
    repr : Coalgebra.Repr R a
    ⊢ Eq (repr.index.sum fun i => TensorProduct.tmul R (repr.left i) (CoalgebraStr …
  -/
  simpa [← repr.eq, map_sum] using congr($(lTensor_counit_comp_comul (R := R) (A := A)) a)
  /-
    🎉 no goals
  -/


@[simp]
lemma sum_tmul_tmul_eq {a : A} (repr : Repr R a)
    (a₁ : (i : repr.ι) → Repr R (repr.left i)) (a₂ : (i : repr.ι) → Repr R (repr.right i)) :
    ∑ i in repr.index, ∑ j in (a₁ i).index,
      (a₁ i).left j ⊗ₜ[R] (a₁ i).right j ⊗ₜ[R] repr.right i
      = ∑ i in repr.index, ∑ j in (a₂ i).index,
      repr.left i ⊗ₜ[R] (a₂ i).left j ⊗ₜ[R] (a₂ i).right j := by
  simpa [(a₂ _).eq, ← (a₁ _).eq, ← TensorProduct.tmul_sum,
    TensorProduct.sum_tmul, ← repr.eq] using congr($(coassoc (R := R)) a)


@[simp]
theorem sum_counit_tmul_map_eq {B : Type*} [AddCommMonoid B] [Module R B]
    {F : Type*} [FunLike F A B] [LinearMapClass F R A B] (f : F) (a : A) {repr : Repr R a} :
    ∑ i in repr.index, counit (R := R) (repr.left i) ⊗ₜ f (repr.right i) = 1 ⊗ₜ[R] f a := by
  /-
    R : Type u
    A : Type v
    inst✝⁷ : CommSemiring R
    inst✝⁶ : AddCommMonoid A
    inst✝⁵ : Module R A
    inst✝⁴ : Coalgebra R A
    B : Type u_1
    inst✝³ : AddCommMonoid B
    inst✝² : Module R B
    F : Type u_2
    inst✝¹ : FunLike F A B
    inst✝ : LinearMapClass F R A B
    f : F
    a : A
    repr : Coalgebra.Repr R a
    ⊢ Eq (repr.index.sum fun i => TensorProduct.tmul R (CoalgebraStruct.counit (re …
  -/
  have := sum_counit_tmul_eq repr
  /-
    R : Type u
    A : Type v
    inst✝⁷ : CommSemiring R
    inst✝⁶ : AddCommMonoid A
    inst✝⁵ : Module R A
    inst✝⁴ : Coalgebra R A
    B : Type u_1
    inst✝³ : AddCommMonoid B
    inst✝² : Module R B
    F : Type u_2
    inst✝¹ : FunLike F A B
    inst✝ : LinearMapClass F R A B
    f : F
    a : A
    repr : Coalgebra.Repr R a
    this : Eq (repr.index.sum fun i => TensorProduct.tmul R (CoalgebraStruct.couni …
    ⊢ Eq (repr.index.sum fun i => TensorProduct.tmul R (CoalgebraStruct.counit (re …
  -/
  apply_fun LinearMap.lTensor R (f : A →ₗ[R] B) at this
  /-
    R : Type u
    A : Type v
    inst✝⁷ : CommSemiring R
    inst✝⁶ : AddCommMonoid A
    inst✝⁵ : Module R A
    inst✝⁴ : Coalgebra R A
    B : Type u_1
    inst✝³ : AddCommMonoid B
    inst✝² : Module R B
    F : Type u_2
    inst✝¹ : FunLike F A B
    inst✝ : LinearMapClass F R A B
    f : F
    a : A
    repr : Coalgebra.Repr R a
    this : Eq ((LinearMap.lTensor R ↑f) (repr.index.sum fun i => TensorProduct.tmu …
    ⊢ Eq (repr.index.sum fun i => TensorProduct.tmul R (CoalgebraStruct.counit (re …
  -/
  simp_all only [map_sum, LinearMap.lTensor_tmul, LinearMap.coe_coe]
  /-
    🎉 no goals
  -/


@[simp]
theorem sum_map_tmul_counit_eq {B : Type*} [AddCommMonoid B] [Module R B]
    {F : Type*} [FunLike F A B] [LinearMapClass F R A B] (f : F) (a : A) {repr : Repr R a} :
    ∑ i in repr.index, f (repr.left i) ⊗ₜ counit (R := R) (repr.right i) = f a ⊗ₜ[R] 1 := by
  /-
    R : Type u
    A : Type v
    inst✝⁷ : CommSemiring R
    inst✝⁶ : AddCommMonoid A
    inst✝⁵ : Module R A
    inst✝⁴ : Coalgebra R A
    B : Type u_1
    inst✝³ : AddCommMonoid B
    inst✝² : Module R B
    F : Type u_2
    inst✝¹ : FunLike F A B
    inst✝ : LinearMapClass F R A B
    f : F
    a : A
    repr : Coalgebra.Repr R a
    ⊢ Eq (repr.index.sum fun i => TensorProduct.tmul R (f (repr.left i)) (Coalgebr …
  -/
  have := sum_tmul_counit_eq repr
  /-
    R : Type u
    A : Type v
    inst✝⁷ : CommSemiring R
    inst✝⁶ : AddCommMonoid A
    inst✝⁵ : Module R A
    inst✝⁴ : Coalgebra R A
    B : Type u_1
    inst✝³ : AddCommMonoid B
    inst✝² : Module R B
    F : Type u_2
    inst✝¹ : FunLike F A B
    inst✝ : LinearMapClass F R A B
    f : F
    a : A
    repr : Coalgebra.Repr R a
    this : Eq (repr.index.sum fun i => TensorProduct.tmul R (repr.left i) (Coalgeb …
    ⊢ Eq (repr.index.sum fun i => TensorProduct.tmul R (f (repr.left i)) (Coalgebr …
  -/
  apply_fun LinearMap.rTensor R (f : A →ₗ[R] B) at this
  /-
    R : Type u
    A : Type v
    inst✝⁷ : CommSemiring R
    inst✝⁶ : AddCommMonoid A
    inst✝⁵ : Module R A
    inst✝⁴ : Coalgebra R A
    B : Type u_1
    inst✝³ : AddCommMonoid B
    inst✝² : Module R B
    F : Type u_2
    inst✝¹ : FunLike F A B
    inst✝ : LinearMapClass F R A B
    f : F
    a : A
    repr : Coalgebra.Repr R a
    this : Eq ((LinearMap.rTensor R ↑f) (repr.index.sum fun i => TensorProduct.tmu …
    ⊢ Eq (repr.index.sum fun i => TensorProduct.tmul R (f (repr.left i)) (Coalgebr …
  -/
  simp_all only [map_sum, LinearMap.rTensor_tmul, LinearMap.coe_coe]
  /-
    🎉 no goals
  -/


@[simp]
theorem sum_map_tmul_tmul_eq {B : Type*} [AddCommMonoid B] [Module R B]
    {F : Type*} [FunLike F A B] [LinearMapClass F R A B] (f g h : F) (a : A) {repr : Repr R a}
    {a₁ : (i : repr.ι) → Repr R (repr.left i)} {a₂ : (i : repr.ι) → Repr R (repr.right i)} :
    ∑ i in repr.index, ∑ j in (a₂ i).index,
      f (repr.left i) ⊗ₜ (g ((a₂ i).left j) ⊗ₜ h ((a₂ i).right j)) =
    ∑ i in repr.index, ∑ j in (a₁ i).index,
      f ((a₁ i).left j) ⊗ₜ[R] (g ((a₁ i).right j) ⊗ₜ[R] h (repr.right i)) := by
  /-
    R : Type u
    A : Type v
    inst✝⁷ : CommSemiring R
    inst✝⁶ : AddCommMonoid A
    inst✝⁵ : Module R A
    inst✝⁴ : Coalgebra R A
    B : Type u_1
    inst✝³ : AddCommMonoid B
    inst✝² : Module R B
    F : Type u_2
    inst✝¹ : FunLike F A B
    inst✝ : LinearMapClass F R A B
    f g h : F
    a : A
    repr : Coalgebra.Repr R a
    a₁ : (i : repr.ι) → Coalgebra.Repr R (repr.left i)
    a₂ : (i : repr.ι) → Coalgebra.Repr R (repr.right i)
    ⊢ Eq (repr.index.sum fun i => (a₂ i).index.sum fun j => TensorProduct.tmul R ( …
  -/
  have := sum_tmul_tmul_eq repr a₁ a₂
  apply_fun TensorProduct.map (f : A →ₗ[R] B)
    (TensorProduct.map (g : A →ₗ[R] B) (h : A →ₗ[R] B)) at this
  /-
    R : Type u
    A : Type v
    inst✝⁷ : CommSemiring R
    inst✝⁶ : AddCommMonoid A
    inst✝⁵ : Module R A
    inst✝⁴ : Coalgebra R A
    B : Type u_1
    inst✝³ : AddCommMonoid B
    inst✝² : Module R B
    F : Type u_2
    inst✝¹ : FunLike F A B
    inst✝ : LinearMapClass F R A B
    f g h : F
    a : A
    repr : Coalgebra.Repr R a
    a₁ : (i : repr.ι) → Coalgebra.Repr R (repr.left i)
    a₂ : (i : repr.ι) → Coalgebra.Repr R (repr.right i)
    this : Eq ((TensorProduct.map (↑f) (TensorProduct.map ↑g ↑h)) (repr.index.sum  …
    ⊢ Eq (repr.index.sum fun i => (a₂ i).index.sum fun j => TensorProduct.tmul R ( …
  -/
  simp_all only [map_sum, TensorProduct.map_tmul, LinearMap.coe_coe]
  /-
    🎉 no goals
  -/


/-- Every commutative (semi)ring is a coalgebra over itself, with `Δ r = 1 ⊗ₜ r`. -/
instance toCoalgebra : Coalgebra R R where
  comul := (TensorProduct.mk R R R) 1
  counit := .id
  coassoc := rfl
                                  /-
                                    R : Type u
                                    inst✝ : CommSemiring R
                                    ⊢ Eq ((LinearMap.rTensor R CoalgebraStruct.counit).comp CoalgebraStruct.comul) …
                                  -/
  rTensor_counit_comp_comul := by ext; rfl
                                       /-
                                         🎉 no goals
                                       -/
                                  /-
                                    R : Type u
                                    inst✝ : CommSemiring R
                                    ⊢ Eq ((LinearMap.lTensor R CoalgebraStruct.counit).comp CoalgebraStruct.comul) …
                                  -/
  lTensor_counit_comp_comul := by ext; rfl
                                       /-
                                         🎉 no goals
                                       -/


@[simp]
theorem comul_apply (r : R) : comul r = 1 ⊗ₜ[R] r := rfl


@[simp]
theorem counit_apply (r : R) : counit r = r := rfl


instance instCoalgebraStruct : CoalgebraStruct R (A × B) where
  comul := .coprod
    (TensorProduct.map (.inl R A B) (.inl R A B) ∘ₗ comul)
    (TensorProduct.map (.inr R A B) (.inr R A B) ∘ₗ comul)
  counit := .coprod counit counit


@[simp]
theorem comul_apply (r : A × B) :
    comul r =
      TensorProduct.map (.inl R A B) (.inl R A B) (comul r.1) +
      TensorProduct.map (.inr R A B) (.inr R A B) (comul r.2) := rfl


@[simp]
theorem counit_apply (r : A × B) : (counit r : R) = counit r.1 + counit r.2 := rfl


theorem comul_comp_inl :
    comul ∘ₗ inl R A B = TensorProduct.map (.inl R A B) (.inl R A B) ∘ₗ comul := by
  /-
    R : Type u
    A : Type v
    B : Type w
    inst✝⁶ : CommSemiring R
    inst✝⁵ : AddCommMonoid A
    inst✝⁴ : AddCommMonoid B
    inst✝³ : Module R A
    inst✝² : Module R B
    inst✝¹ : Coalgebra R A
    inst✝ : Coalgebra R B
    ⊢ Eq (CoalgebraStruct.comul.comp (LinearMap.inl R A B)) ((TensorProduct.map (L …
  -/
  ext; simp
       /-
         🎉 no goals
       -/


theorem comul_comp_inr :
    comul ∘ₗ inr R A B = TensorProduct.map (.inr R A B) (.inr R A B) ∘ₗ comul := by
  /-
    R : Type u
    A : Type v
    B : Type w
    inst✝⁶ : CommSemiring R
    inst✝⁵ : AddCommMonoid A
    inst✝⁴ : AddCommMonoid B
    inst✝³ : Module R A
    inst✝² : Module R B
    inst✝¹ : Coalgebra R A
    inst✝ : Coalgebra R B
    ⊢ Eq (CoalgebraStruct.comul.comp (LinearMap.inr R A B)) ((TensorProduct.map (L …
  -/
  ext; simp
       /-
         🎉 no goals
       -/


theorem comul_comp_fst :
    comul ∘ₗ .fst R A B = TensorProduct.map (.fst R A B) (.fst R A B) ∘ₗ comul := by
  /-
    R : Type u
    A : Type v
    B : Type w
    inst✝⁶ : CommSemiring R
    inst✝⁵ : AddCommMonoid A
    inst✝⁴ : AddCommMonoid B
    inst✝³ : Module R A
    inst✝² : Module R B
    inst✝¹ : Coalgebra R A
    inst✝ : Coalgebra R B
    ⊢ Eq (CoalgebraStruct.comul.comp (LinearMap.fst R A B)) ((TensorProduct.map (L …
  -/
  ext : 1
  · rw [comp_assoc, fst_comp_inl, comp_id, comp_assoc, comul_comp_inl, ← comp_assoc,
      ← TensorProduct.map_comp, fst_comp_inl, TensorProduct.map_id, id_comp]
  · rw [comp_assoc, fst_comp_inr, comp_zero, comp_assoc, comul_comp_inr, ← comp_assoc,
      ← TensorProduct.map_comp, fst_comp_inr, TensorProduct.map_zero_left, zero_comp]


theorem comul_comp_snd :
    comul ∘ₗ .snd R A B = TensorProduct.map (.snd R A B) (.snd R A B) ∘ₗ comul := by
  /-
    R : Type u
    A : Type v
    B : Type w
    inst✝⁶ : CommSemiring R
    inst✝⁵ : AddCommMonoid A
    inst✝⁴ : AddCommMonoid B
    inst✝³ : Module R A
    inst✝² : Module R B
    inst✝¹ : Coalgebra R A
    inst✝ : Coalgebra R B
    ⊢ Eq (CoalgebraStruct.comul.comp (LinearMap.snd R A B)) ((TensorProduct.map (L …
  -/
  ext : 1
  · rw [comp_assoc, snd_comp_inl, comp_zero, comp_assoc, comul_comp_inl, ← comp_assoc,
      ← TensorProduct.map_comp, snd_comp_inl, TensorProduct.map_zero_left, zero_comp]
  · rw [comp_assoc, snd_comp_inr, comp_id, comp_assoc, comul_comp_inr, ← comp_assoc,
      ← TensorProduct.map_comp, snd_comp_inr, TensorProduct.map_id, id_comp]


                                                                     /-
                                                                       R : Type u
                                                                       A : Type v
                                                                       B : Type w
                                                                       inst✝⁶ : CommSemiring R
                                                                       inst✝⁵ : AddCommMonoid A
                                                                       inst✝⁴ : AddCommMonoid B
                                                                       inst✝³ : Module R A
                                                                       inst✝² : Module R B
                                                                       inst✝¹ : Coalgebra R A
                                                                       inst✝ : Coalgebra R B
                                                                       ⊢ Eq (CoalgebraStruct.counit.comp (LinearMap.inr R A B)) CoalgebraStruct.counit
                                                                     -/
@[simp] theorem counit_comp_inr : counit ∘ₗ inr R A B = counit := by ext; simp
                                                                          /-
                                                                            🎉 no goals
                                                                          -/


                                                                     /-
                                                                       R : Type u
                                                                       A : Type v
                                                                       B : Type w
                                                                       inst✝⁶ : CommSemiring R
                                                                       inst✝⁵ : AddCommMonoid A
                                                                       inst✝⁴ : AddCommMonoid B
                                                                       inst✝³ : Module R A
                                                                       inst✝² : Module R B
                                                                       inst✝¹ : Coalgebra R A
                                                                       inst✝ : Coalgebra R B
                                                                       ⊢ Eq (CoalgebraStruct.counit.comp (LinearMap.inl R A B)) CoalgebraStruct.counit
                                                                     -/
@[simp] theorem counit_comp_inl : counit ∘ₗ inl R A B = counit := by ext; simp
                                                                          /-
                                                                            🎉 no goals
                                                                          -/


instance instCoalgebra : Coalgebra R (A × B) where
  rTensor_counit_comp_comul := by
    /-
      R : Type u
      A : Type v
      B : Type w
      inst✝⁶ : CommSemiring R
      inst✝⁵ : AddCommMonoid A
      inst✝⁴ : AddCommMonoid B
      inst✝³ : Module R A
      inst✝² : Module R B
      inst✝¹ : Coalgebra R A
      inst✝ : Coalgebra R B
      ⊢ Eq ((LinearMap.rTensor (Prod A B) CoalgebraStruct.counit).comp CoalgebraStru …
    -/
    ext : 1
    · rw [comp_assoc, comul_comp_inl, ← comp_assoc, rTensor_comp_map, counit_comp_inl,
        ← lTensor_comp_rTensor, comp_assoc, rTensor_counit_comp_comul, lTensor_comp_mk]
    · rw [comp_assoc, comul_comp_inr, ← comp_assoc, rTensor_comp_map, counit_comp_inr,
        ← lTensor_comp_rTensor, comp_assoc, rTensor_counit_comp_comul, lTensor_comp_mk]
  lTensor_counit_comp_comul := by
    /-
      R : Type u
      A : Type v
      B : Type w
      inst✝⁶ : CommSemiring R
      inst✝⁵ : AddCommMonoid A
      inst✝⁴ : AddCommMonoid B
      inst✝³ : Module R A
      inst✝² : Module R B
      inst✝¹ : Coalgebra R A
      inst✝ : Coalgebra R B
      ⊢ Eq ((LinearMap.lTensor (Prod A B) CoalgebraStruct.counit).comp CoalgebraStru …
    -/
    ext : 1
    · rw [comp_assoc, comul_comp_inl, ← comp_assoc, lTensor_comp_map, counit_comp_inl,
        ← rTensor_comp_lTensor, comp_assoc, lTensor_counit_comp_comul, rTensor_comp_flip_mk]
    · rw [comp_assoc, comul_comp_inr, ← comp_assoc, lTensor_comp_map, counit_comp_inr,
    /-
      R : Type u
      A : Type v
      B : Type w
      inst✝⁶ : CommSemiring R
      inst✝⁵ : AddCommMonoid A
      inst✝⁴ : AddCommMonoid B
      inst✝³ : Module R A
      inst✝² : Module R B
      inst✝¹ : Coalgebra R A
      inst✝ : Coalgebra R B
      ⊢ Eq ((↑(TensorProduct.assoc R (Prod A B) (Prod A B) (Prod A B))).comp ((Linea …
    -/
        ← rTensor_comp_lTensor, comp_assoc, lTensor_counit_comp_comul, rTensor_comp_flip_mk]
    /-
      R : Type u
      A : Type v
      B : Type w
      inst✝⁶ : CommSemiring R
      inst✝⁵ : AddCommMonoid A
      inst✝⁴ : AddCommMonoid B
      inst✝³ : Module R A
      inst✝² : Module R B
      inst✝¹ : Coalgebra R A
      inst✝ : Coalgebra R B
      ⊢ Eq ((↑(TensorProduct.assoc R (Prod A B) (Prod A B) (Prod A B))).comp ((Linea …
    -/
  coassoc := by
      /-
        case hl.h
        R : Type u
        A : Type v
        B : Type w
        inst✝⁶ : CommSemiring R
        inst✝⁵ : AddCommMonoid A
        inst✝⁴ : AddCommMonoid B
        inst✝³ : Module R A
        inst✝² : Module R B
        inst✝¹ : Coalgebra R A
        inst✝ : Coalgebra R B
        x : A
        ⊢ Eq ((TensorProduct.assoc R (Prod A B) (Prod A B) (Prod A B)) ((LinearMap.rTe …
      -/
    dsimp only [instCoalgebraStruct]
    ext x : 2 <;> dsimp only [comp_apply, LinearEquiv.coe_coe, coe_inl, coe_inr, coprod_apply]
    · simp only [map_zero, add_zero]
      simp_rw [← comp_apply, ← comp_assoc, rTensor_comp_map, lTensor_comp_map, coprod_inl,
      /-
        case hr.h
        R : Type u
        A : Type v
        B : Type w
        inst✝⁶ : CommSemiring R
        inst✝⁵ : AddCommMonoid A
        inst✝⁴ : AddCommMonoid B
        inst✝³ : Module R A
        inst✝² : Module R B
        inst✝¹ : Coalgebra R A
        inst✝ : Coalgebra R B
        x : B
        ⊢ Eq ((TensorProduct.assoc R (Prod A B) (Prod A B) (Prod A B)) ((LinearMap.rTe …
      -/
        ← map_comp_rTensor, ← map_comp_lTensor, comp_assoc, ← coassoc, ← comp_assoc,
        TensorProduct.map_map_comp_assoc_eq, comp_apply, LinearEquiv.coe_coe]
    · simp only [map_zero, zero_add]
      simp_rw [← comp_apply, ← comp_assoc, rTensor_comp_map, lTensor_comp_map, coprod_inr,
        ← map_comp_rTensor, ← map_comp_lTensor, comp_assoc, ← coassoc, ← comp_assoc,
        TensorProduct.map_map_comp_assoc_eq, comp_apply, LinearEquiv.coe_coe]


instance instCoalgebraStruct : CoalgebraStruct R (Π₀ i, A i) where
  comul := DFinsupp.lsum R fun i =>
    TensorProduct.map (DFinsupp.lsingle i) (DFinsupp.lsingle i) ∘ₗ comul
  counit := DFinsupp.lsum R fun _ => counit


@[simp]
theorem comul_single (i : ι) (a : A i) :
    comul (R := R) (DFinsupp.single i a) =
      (TensorProduct.map (DFinsupp.lsingle i) (DFinsupp.lsingle i) : _ →ₗ[R] _) (comul a) :=
  lsum_single _ _ _ _


@[simp]
theorem counit_single (i : ι) (a : A i) : counit (DFinsupp.single i a) = counit (R := R) a :=
  lsum_single _ _ _ _


theorem comul_comp_lsingle (i : ι) :
    comul ∘ₗ (lsingle i : A i →ₗ[R] _) = TensorProduct.map (lsingle i) (lsingle i) ∘ₗ comul := by
  /-
    R : Type u
    ι : Type v
    A : ι → Type w
    inst✝⁴ : DecidableEq ι
    inst✝³ : CommSemiring R
    inst✝² : (i : ι) → AddCommMonoid (A i)
    inst✝¹ : (i : ι) → Module R (A i)
    inst✝ : (i : ι) → Coalgebra R (A i)
    i : ι
    ⊢ Eq (CoalgebraStruct.comul.comp (DFinsupp.lsingle i)) ((TensorProduct.map (DF …
  -/
  ext; simp
       /-
         🎉 no goals
       -/


theorem comul_comp_lapply (i : ι) :
    comul ∘ₗ (lapply i : _ →ₗ[R] A i) = TensorProduct.map (lapply i) (lapply i) ∘ₗ comul := by
  /-
    R : Type u
    ι : Type v
    A : ι → Type w
    inst✝⁴ : DecidableEq ι
    inst✝³ : CommSemiring R
    inst✝² : (i : ι) → AddCommMonoid (A i)
    inst✝¹ : (i : ι) → Module R (A i)
    inst✝ : (i : ι) → Coalgebra R (A i)
    i : ι
    ⊢ Eq (CoalgebraStruct.comul.comp (DFinsupp.lapply i)) ((TensorProduct.map (DFi …
  -/
  ext j : 1
  /-
    case h
    R : Type u
    ι : Type v
    A : ι → Type w
    inst✝⁴ : DecidableEq ι
    inst✝³ : CommSemiring R
    inst✝² : (i : ι) → AddCommMonoid (A i)
    inst✝¹ : (i : ι) → Module R (A i)
    inst✝ : (i : ι) → Coalgebra R (A i)
    i j : ι
    ⊢ Eq ((CoalgebraStruct.comul.comp (DFinsupp.lapply i)).comp (DFinsupp.lsingle  …
  -/
  conv_rhs => rw [comp_assoc, comul_comp_lsingle, ← comp_assoc, ← TensorProduct.map_comp]
  /-
    case h
    R : Type u
    ι : Type v
    A : ι → Type w
    inst✝⁴ : DecidableEq ι
    inst✝³ : CommSemiring R
    inst✝² : (i : ι) → AddCommMonoid (A i)
    inst✝¹ : (i : ι) → Module R (A i)
    inst✝ : (i : ι) → Coalgebra R (A i)
    i j : ι
    ⊢ Eq ((CoalgebraStruct.comul.comp (DFinsupp.lapply i)).comp (DFinsupp.lsingle  …
  -/
  obtain rfl | hij := eq_or_ne i j
    /-
      case h.inl
      R : Type u
      ι : Type v
      A : ι → Type w
      inst✝⁴ : DecidableEq ι
      inst✝³ : CommSemiring R
      inst✝² : (i : ι) → AddCommMonoid (A i)
      inst✝¹ : (i : ι) → Module R (A i)
      inst✝ : (i : ι) → Coalgebra R (A i)
      i : ι
      ⊢ Eq ((CoalgebraStruct.comul.comp (DFinsupp.lapply i)).comp (DFinsupp.lsingle  …
    -/
  · rw [comp_assoc, lapply_comp_lsingle_same, comp_id,  TensorProduct.map_id, id_comp]
    /-
      🎉 no goals
    -/
  · rw [comp_assoc, lapply_comp_lsingle_of_ne _ _ hij, comp_zero, TensorProduct.map_zero_left,
      zero_comp]


@[simp] theorem counit_comp_lsingle (i : ι) : counit ∘ₗ (lsingle i : A i →ₗ[R] _) = counit := by
  /-
    R : Type u
    ι : Type v
    A : ι → Type w
    inst✝⁴ : DecidableEq ι
    inst✝³ : CommSemiring R
    inst✝² : (i : ι) → AddCommMonoid (A i)
    inst✝¹ : (i : ι) → Module R (A i)
    inst✝ : (i : ι) → Coalgebra R (A i)
    i : ι
    ⊢ Eq (CoalgebraStruct.counit.comp (DFinsupp.lsingle i)) CoalgebraStruct.counit
  -/
  ext; simp
       /-
         🎉 no goals
       -/


/-- The `R`-module whose elements are dependent functions `(i : ι) → A i` which are zero on all but
finitely many elements of `ι` has a coalgebra structure.

The coproduct `Δ` is given by `Δ(fᵢ a) = fᵢ a₁ ⊗ fᵢ a₂` where `Δ(a) = a₁ ⊗ a₂` and the counit `ε`
by `ε(fᵢ a) = ε(a)`, where `fᵢ a` is the function sending `i` to `a` and all other elements of `ι`
to zero. -/
instance instCoalgebra : Coalgebra R (Π₀ i, A i) where
  rTensor_counit_comp_comul := by
    /-
      R : Type u
      ι : Type v
      A : ι → Type w
      inst✝⁴ : DecidableEq ι
      inst✝³ : CommSemiring R
      inst✝² : (i : ι) → AddCommMonoid (A i)
      inst✝¹ : (i : ι) → Module R (A i)
      inst✝ : (i : ι) → Coalgebra R (A i)
      ⊢ Eq ((LinearMap.rTensor (DFinsupp fun i => A i) CoalgebraStruct.counit).comp  …
    -/
    ext : 1
    rw [comp_assoc, comul_comp_lsingle, ← comp_assoc, rTensor_comp_map, counit_comp_lsingle,
      ← lTensor_comp_rTensor, comp_assoc, rTensor_counit_comp_comul, lTensor_comp_mk]
  lTensor_counit_comp_comul := by
    /-
      R : Type u
      ι : Type v
      A : ι → Type w
      inst✝⁴ : DecidableEq ι
      inst✝³ : CommSemiring R
      inst✝² : (i : ι) → AddCommMonoid (A i)
      inst✝¹ : (i : ι) → Module R (A i)
      inst✝ : (i : ι) → Coalgebra R (A i)
      ⊢ Eq ((LinearMap.lTensor (DFinsupp fun i => A i) CoalgebraStruct.counit).comp  …
    -/
    ext : 1
    rw [comp_assoc, comul_comp_lsingle, ← comp_assoc, lTensor_comp_map, counit_comp_lsingle,
    /-
      R : Type u
      ι : Type v
      A : ι → Type w
      inst✝⁴ : DecidableEq ι
      inst✝³ : CommSemiring R
      inst✝² : (i : ι) → AddCommMonoid (A i)
      inst✝¹ : (i : ι) → Module R (A i)
      inst✝ : (i : ι) → Coalgebra R (A i)
      ⊢ Eq ((↑(TensorProduct.assoc R (DFinsupp fun i => A i) (DFinsupp fun i => A i) …
    -/
      ← rTensor_comp_lTensor, comp_assoc, lTensor_counit_comp_comul, rTensor_comp_flip_mk]
  coassoc := by
    ext i : 1
    simp_rw [comp_assoc, comul_comp_lsingle, ← comp_assoc, lTensor_comp_map, comul_comp_lsingle,
      comp_assoc, ← comp_assoc comul, rTensor_comp_map, comul_comp_lsingle, ← map_comp_rTensor,
      ← map_comp_lTensor, comp_assoc, ← coassoc, ← comp_assoc comul, ← comp_assoc,
        TensorProduct.map_map_comp_assoc_eq]


instance instCoalgebraStruct : CoalgebraStruct R (ι →₀ A) where
  comul := Finsupp.lsum R fun i =>
    TensorProduct.map (Finsupp.lsingle i) (Finsupp.lsingle i) ∘ₗ comul
  counit := Finsupp.lsum R fun _ => counit


@[simp]
theorem comul_single (i : ι) (a : A) :
    comul (R := R) (Finsupp.single i a) =
      (TensorProduct.map (Finsupp.lsingle i) (Finsupp.lsingle i) : _ →ₗ[R] _) (comul a) :=
  lsum_single _ _ _ _


@[simp]
theorem counit_single (i : ι) (a : A) : counit (Finsupp.single i a) = counit (R := R) a :=
  lsum_single _ _ _ _


theorem comul_comp_lsingle (i : ι) :
    comul ∘ₗ (lsingle i : A →ₗ[R] _) = TensorProduct.map (lsingle i) (lsingle i) ∘ₗ comul := by
  /-
    R : Type u
    ι : Type v
    A : Type w
    inst✝³ : CommSemiring R
    inst✝² : AddCommMonoid A
    inst✝¹ : Module R A
    inst✝ : Coalgebra R A
    i : ι
    ⊢ Eq (CoalgebraStruct.comul.comp (Finsupp.lsingle i)) ((TensorProduct.map (Fin …
  -/
  ext; simp
       /-
         🎉 no goals
       -/


theorem comul_comp_lapply (i : ι) :
    comul ∘ₗ (lapply i : _ →ₗ[R] A) = TensorProduct.map (lapply i) (lapply i) ∘ₗ comul := by
  /-
    R : Type u
    ι : Type v
    A : Type w
    inst✝³ : CommSemiring R
    inst✝² : AddCommMonoid A
    inst✝¹ : Module R A
    inst✝ : Coalgebra R A
    i : ι
    ⊢ Eq (CoalgebraStruct.comul.comp (Finsupp.lapply i)) ((TensorProduct.map (Fins …
  -/
  ext j : 1
  /-
    case h
    R : Type u
    ι : Type v
    A : Type w
    inst✝³ : CommSemiring R
    inst✝² : AddCommMonoid A
    inst✝¹ : Module R A
    inst✝ : Coalgebra R A
    i j : ι
    ⊢ Eq ((CoalgebraStruct.comul.comp (Finsupp.lapply i)).comp (Finsupp.lsingle j) …
  -/
  conv_rhs => rw [comp_assoc, comul_comp_lsingle, ← comp_assoc, ← TensorProduct.map_comp]
  /-
    case h
    R : Type u
    ι : Type v
    A : Type w
    inst✝³ : CommSemiring R
    inst✝² : AddCommMonoid A
    inst✝¹ : Module R A
    inst✝ : Coalgebra R A
    i j : ι
    ⊢ Eq ((CoalgebraStruct.comul.comp (Finsupp.lapply i)).comp (Finsupp.lsingle j) …
  -/
  obtain rfl | hij := eq_or_ne i j
    /-
      case h.inl
      R : Type u
      ι : Type v
      A : Type w
      inst✝³ : CommSemiring R
      inst✝² : AddCommMonoid A
      inst✝¹ : Module R A
      inst✝ : Coalgebra R A
      i : ι
      ⊢ Eq ((CoalgebraStruct.comul.comp (Finsupp.lapply i)).comp (Finsupp.lsingle i) …
    -/
  · rw [comp_assoc, lapply_comp_lsingle_same, comp_id,  TensorProduct.map_id, id_comp]
    /-
      🎉 no goals
    -/
  · rw [comp_assoc, lapply_comp_lsingle_of_ne _ _ hij, comp_zero, TensorProduct.map_zero_left,
      zero_comp]


@[simp] theorem counit_comp_lsingle (i : ι) : counit ∘ₗ (lsingle i : A →ₗ[R] _) = counit := by
  /-
    R : Type u
    ι : Type v
    A : Type w
    inst✝³ : CommSemiring R
    inst✝² : AddCommMonoid A
    inst✝¹ : Module R A
    inst✝ : Coalgebra R A
    i : ι
    ⊢ Eq (CoalgebraStruct.counit.comp (Finsupp.lsingle i)) CoalgebraStruct.counit
  -/
  ext; simp
       /-
         🎉 no goals
       -/


/-- The `R`-module whose elements are functions `ι → A` which are zero on all but finitely many
elements of `ι` has a coalgebra structure. The coproduct `Δ` is given by `Δ(fᵢ a) = fᵢ a₁ ⊗ fᵢ a₂`
where `Δ(a) = a₁ ⊗ a₂` and the counit `ε` by `ε(fᵢ a) = ε(a)`, where `fᵢ a` is the function sending
`i` to `a` and all other elements of `ι` to zero. -/
instance instCoalgebra : Coalgebra R (ι →₀ A) where
  rTensor_counit_comp_comul := by
    /-
      R : Type u
      ι : Type v
      A : Type w
      inst✝³ : CommSemiring R
      inst✝² : AddCommMonoid A
      inst✝¹ : Module R A
      inst✝ : Coalgebra R A
      ⊢ Eq ((LinearMap.rTensor (Finsupp ι A) CoalgebraStruct.counit).comp CoalgebraS …
    -/
    ext : 1
    rw [comp_assoc, comul_comp_lsingle, ← comp_assoc, rTensor_comp_map, counit_comp_lsingle,
      ← lTensor_comp_rTensor, comp_assoc, rTensor_counit_comp_comul, lTensor_comp_mk]
  lTensor_counit_comp_comul := by
    /-
      R : Type u
      ι : Type v
      A : Type w
      inst✝³ : CommSemiring R
      inst✝² : AddCommMonoid A
      inst✝¹ : Module R A
      inst✝ : Coalgebra R A
      ⊢ Eq ((LinearMap.lTensor (Finsupp ι A) CoalgebraStruct.counit).comp CoalgebraS …
    -/
    ext : 1
    rw [comp_assoc, comul_comp_lsingle, ← comp_assoc, lTensor_comp_map, counit_comp_lsingle,
    /-
      R : Type u
      ι : Type v
      A : Type w
      inst✝³ : CommSemiring R
      inst✝² : AddCommMonoid A
      inst✝¹ : Module R A
      inst✝ : Coalgebra R A
      ⊢ Eq ((↑(TensorProduct.assoc R (Finsupp ι A) (Finsupp ι A) (Finsupp ι A))).com …
    -/
      ← rTensor_comp_lTensor, comp_assoc, lTensor_counit_comp_comul, rTensor_comp_flip_mk]
  coassoc := by
    ext i : 1
    simp_rw [comp_assoc, comul_comp_lsingle, ← comp_assoc, lTensor_comp_map, comul_comp_lsingle,
      comp_assoc, ← comp_assoc comul, rTensor_comp_map, comul_comp_lsingle, ← map_comp_rTensor,
      ← map_comp_lTensor, comp_assoc, ← coassoc, ← comp_assoc comul, ← comp_assoc,
        TensorProduct.map_map_comp_assoc_eq]


/-- See `Mathlib.RingTheory.Coalgebra.TensorProduct` for the `Coalgebra` instance. -/
@[simps] instance instCoalgebraStruct :
    CoalgebraStruct R (A ⊗[R] B) where
  comul := TensorProduct.tensorTensorTensorComm R A A B B ∘ₗ TensorProduct.map comul comul
  counit := LinearMap.mul' R R ∘ₗ TensorProduct.map counit counit


