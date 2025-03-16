/-- A graded version of `DistribMulAction`. -/
class GdistribMulAction [AddMonoid ιA] [VAdd ιA ιB] [GMonoid A] [∀ i, AddMonoid (M i)]
    extends GMulAction A M where
  smul_add {i j} (a : A i) (b c : M j) : smul a (b + c) = smul a b + smul a c
  smul_zero {i j} (a : A i) : smul a (0 : M j) = 0


/-- A graded version of `Module`. -/
class Gmodule [AddMonoid ιA] [VAdd ιA ιB] [∀ i, AddMonoid (A i)] [∀ i, AddMonoid (M i)] [GMonoid A]
    extends GdistribMulAction A M where
  add_smul {i j} (a a' : A i) (b : M j) : smul (a + a') b = smul a b + smul a' b
  zero_smul {i j} (b : M j) : smul (0 : A i) b = 0


/-- A graded version of `Semiring.toModule`. -/
instance GSemiring.toGmodule [AddMonoid ιA] [∀ i : ιA, AddCommMonoid (A i)]
    [h : GSemiring A] : Gmodule A A :=
  { GMonoid.toGMulAction A with
    smul_add := fun _ _ _ => h.mul_add _ _ _
    smul_zero := fun _ => h.mul_zero _
    add_smul := fun _ _ => h.add_mul _ _
    zero_smul := fun _ => h.zero_mul _ }


/-- The piecewise multiplication from the `Mul` instance, as a bundled homomorphism. -/
@[simps]
def gsmulHom [GMonoid A] [Gmodule A M] {i j} : A i →+ M j →+ M (i +ᵥ j) where
  toFun a :=
    { toFun := fun b => GSMul.smul a b
      map_zero' := GdistribMulAction.smul_zero _
      map_add' := GdistribMulAction.smul_add _ }
  map_zero' := AddMonoidHom.ext fun a => Gmodule.zero_smul a
  map_add' _a₁ _a₂ := AddMonoidHom.ext fun _b => Gmodule.add_smul _ _ _


/-- For graded monoid `A` and a graded module `M` over `A`. `Gmodule.smulAddMonoidHom` is the
`⨁ᵢ Aᵢ`-scalar multiplication on `⨁ᵢ Mᵢ` induced by `gsmul_hom`. -/
def smulAddMonoidHom [DecidableEq ιA] [DecidableEq ιB] [GMonoid A] [Gmodule A M] :
    (⨁ i, A i) →+ (⨁ i, M i) →+ ⨁ i, M i :=
  toAddMonoid fun _i =>
    AddMonoidHom.flip <|
      toAddMonoid fun _j => AddMonoidHom.flip <| (of M _).compHom.comp <| gsmulHom A M


instance [DecidableEq ιA] [DecidableEq ιB] [GMonoid A] [Gmodule A M] :
    SMul (⨁ i, A i) (⨁ i, M i) where
  smul x y := smulAddMonoidHom A M x y


@[simp]
theorem smul_def [DecidableEq ιA] [DecidableEq ιB] [GMonoid A] [Gmodule A M]
    (x : ⨁ i, A i) (y : ⨁ i, M i) :
    x • y = smulAddMonoidHom _ _ x y := rfl


@[simp]
theorem smulAddMonoidHom_apply_of_of [DecidableEq ιA] [DecidableEq ιB] [GMonoid A] [Gmodule A M]
    {i j} (x : A i) (y : M j) :
    smulAddMonoidHom A M (DirectSum.of A i x) (of M j y) = of M (i +ᵥ j) (GSMul.smul x y) := by
  /-
    ιA : Type u_1
    ιB : Type u_2
    A : ιA → Type u_3
    M : ιB → Type u_4
    inst✝⁷ : AddMonoid ιA
    inst✝⁶ : VAdd ιA ιB
    inst✝⁵ : (i : ιA) → AddCommMonoid (A i)
    inst✝⁴ : (i : ιB) → AddCommMonoid (M i)
    inst✝³ : DecidableEq ιA
    inst✝² : DecidableEq ιB
    inst✝¹ : GradedMonoid.GMonoid A
    inst✝ : DirectSum.Gmodule A M
    i : ιA
    j : ιB
    x : A i
    y : M j
    ⊢ Eq (((DirectSum.Gmodule.smulAddMonoidHom A M) ((DirectSum.of A i) x)) ((Dire …
  -/
  simp [smulAddMonoidHom]
  /-
    🎉 no goals
  -/


theorem of_smul_of [DecidableEq ιA] [DecidableEq ιB] [GMonoid A] [Gmodule A M]
    {i j} (x : A i) (y : M j) :
                                                                         /-
                                                                           ιA : Type u_1
                                                                           ιB : Type u_2
                                                                           A : ιA → Type u_3
                                                                           M : ιB → Type u_4
                                                                           inst✝⁷ : AddMonoid ιA
                                                                           inst✝⁶ : VAdd ιA ιB
                                                                           inst✝⁵ : (i : ιA) → AddCommMonoid (A i)
                                                                           inst✝⁴ : (i : ιB) → AddCommMonoid (M i)
                                                                           inst✝³ : DecidableEq ιA
                                                                           inst✝² : DecidableEq ιB
                                                                           inst✝¹ : GradedMonoid.GMonoid A
                                                                           inst✝ : DirectSum.Gmodule A M
                                                                           i : ιA
                                                                           j : ιB
                                                                           x : A i
                                                                           y : M j
                                                                           ⊢ Eq (HSMul.hSMul ((DirectSum.of A i) x) ((DirectSum.of M j) y)) ((DirectSum.o …
                                                                         -/
    DirectSum.of A i x • of M j y = of M (i +ᵥ j) (GSMul.smul x y) := by simp
                                                                         /-
                                                                           🎉 no goals
                                                                         -/


private theorem one_smul' [DecidableEq ιA] [DecidableEq ιB] [GMonoid A] [Gmodule A M]
    (x : ⨁ i, M i) :
    (1 : ⨁ i, A i) • x = x := by
  /-
    ιA : Type u_1
    ιB : Type u_2
    A : ιA → Type u_3
    M : ιB → Type u_4
    inst✝⁷ : AddMonoid ιA
    inst✝⁶ : VAdd ιA ιB
    inst✝⁵ : (i : ιA) → AddCommMonoid (A i)
    inst✝⁴ : (i : ιB) → AddCommMonoid (M i)
    inst✝³ : DecidableEq ιA
    inst✝² : DecidableEq ιB
    inst✝¹ : GradedMonoid.GMonoid A
    inst✝ : DirectSum.Gmodule A M
    x : DirectSum ιB fun i => M i
    ⊢ Eq (HSMul.hSMul 1 x) x
  -/
  suffices smulAddMonoidHom A M 1 = AddMonoidHom.id (⨁ i, M i) from DFunLike.congr_fun this x
  /-
    ιA : Type u_1
    ιB : Type u_2
    A : ιA → Type u_3
    M : ιB → Type u_4
    inst✝⁷ : AddMonoid ιA
    inst✝⁶ : VAdd ιA ιB
    inst✝⁵ : (i : ιA) → AddCommMonoid (A i)
    inst✝⁴ : (i : ιB) → AddCommMonoid (M i)
    inst✝³ : DecidableEq ιA
    inst✝² : DecidableEq ιB
    inst✝¹ : GradedMonoid.GMonoid A
    inst✝ : DirectSum.Gmodule A M
    x : DirectSum ιB fun i => M i
    ⊢ Eq ((DirectSum.Gmodule.smulAddMonoidHom A M) 1) (AddMonoidHom.id (DirectSum  …
  -/
  apply DirectSum.addHom_ext; intro i xi
  /-
    case H
    ιA : Type u_1
    ιB : Type u_2
    A : ιA → Type u_3
    M : ιB → Type u_4
    inst✝⁷ : AddMonoid ιA
    inst✝⁶ : VAdd ιA ιB
    inst✝⁵ : (i : ιA) → AddCommMonoid (A i)
    inst✝⁴ : (i : ιB) → AddCommMonoid (M i)
    inst✝³ : DecidableEq ιA
    inst✝² : DecidableEq ιB
    inst✝¹ : GradedMonoid.GMonoid A
    inst✝ : DirectSum.Gmodule A M
    x : DirectSum ιB fun i => M i
    i : ιB
    xi : M i
    ⊢ Eq (((DirectSum.Gmodule.smulAddMonoidHom A M) 1) ((DirectSum.of M i) xi)) (( …
  -/
  rw [show (1 : DirectSum ιA fun i => A i) = (of A 0) GOne.one by rfl]
  /-
    case H
    ιA : Type u_1
    ιB : Type u_2
    A : ιA → Type u_3
    M : ιB → Type u_4
    inst✝⁷ : AddMonoid ιA
    inst✝⁶ : VAdd ιA ιB
    inst✝⁵ : (i : ιA) → AddCommMonoid (A i)
    inst✝⁴ : (i : ιB) → AddCommMonoid (M i)
    inst✝³ : DecidableEq ιA
    inst✝² : DecidableEq ιB
    inst✝¹ : GradedMonoid.GMonoid A
    inst✝ : DirectSum.Gmodule A M
    x : DirectSum ιB fun i => M i
    i : ιB
    xi : M i
    ⊢ Eq (((DirectSum.Gmodule.smulAddMonoidHom A M) ((DirectSum.of A 0) GradedMono …
  -/
  rw [smulAddMonoidHom_apply_of_of]
  /-
    case H
    ιA : Type u_1
    ιB : Type u_2
    A : ιA → Type u_3
    M : ιB → Type u_4
    inst✝⁷ : AddMonoid ιA
    inst✝⁶ : VAdd ιA ιB
    inst✝⁵ : (i : ιA) → AddCommMonoid (A i)
    inst✝⁴ : (i : ιB) → AddCommMonoid (M i)
    inst✝³ : DecidableEq ιA
    inst✝² : DecidableEq ιB
    inst✝¹ : GradedMonoid.GMonoid A
    inst✝ : DirectSum.Gmodule A M
    x : DirectSum ιB fun i => M i
    i : ιB
    xi : M i
    ⊢ Eq ((DirectSum.of M (HVAdd.hVAdd 0 i)) (GradedMonoid.GSMul.smul GradedMonoid …
  -/
  exact DirectSum.of_eq_of_gradedMonoid_eq (one_smul (GradedMonoid A) <| GradedMonoid.mk i xi)
  /-
    🎉 no goals
  -/

-- Porting note: renamed to mul_smul' since DirectSum.Gmodule.mul_smul already exists
-- Almost identical to the proof of `direct_sum.mul_assoc`

private theorem mul_smul' [DecidableEq ιA] [DecidableEq ιB] [GSemiring A] [Gmodule A M]
    (a b : ⨁ i, A i)
    (c : ⨁ i, M i) : (a * b) • c = a • b • c := by
  suffices
    (-- `fun a b c ↦ (a * b) • c` as a bundled hom
              smulAddMonoidHom
              A M).compHom.comp
        (DirectSum.mulHom A) =
      (AddMonoidHom.compHom AddMonoidHom.flipHom <|
          (smulAddMonoidHom A M).flip.compHom.comp <| smulAddMonoidHom A M).flip
    from-- `fun a b c ↦ a • (b • c)` as a bundled hom
      DFunLike.congr_fun (DFunLike.congr_fun (DFunLike.congr_fun this a) b) c
  /-
    ιA : Type u_1
    ιB : Type u_2
    A : ιA → Type u_3
    M : ιB → Type u_4
    inst✝⁷ : AddMonoid ιA
    inst✝⁶ : VAdd ιA ιB
    inst✝⁵ : (i : ιA) → AddCommMonoid (A i)
    inst✝⁴ : (i : ιB) → AddCommMonoid (M i)
    inst✝³ : DecidableEq ιA
    inst✝² : DecidableEq ιB
    inst✝¹ : DirectSum.GSemiring A
    inst✝ : DirectSum.Gmodule A M
    a b : DirectSum ιA fun i => A i
    c : DirectSum ιB fun i => M i
    ⊢ Eq ((AddMonoidHom.compHom (DirectSum.Gmodule.smulAddMonoidHom A M)).comp (Di …
  -/
  ext ai ax bi bx ci cx : 6
  /-
    case H.h.H.h.H.h
    ιA : Type u_1
    ιB : Type u_2
    A : ιA → Type u_3
    M : ιB → Type u_4
    inst✝⁷ : AddMonoid ιA
    inst✝⁶ : VAdd ιA ιB
    inst✝⁵ : (i : ιA) → AddCommMonoid (A i)
    inst✝⁴ : (i : ιB) → AddCommMonoid (M i)
    inst✝³ : DecidableEq ιA
    inst✝² : DecidableEq ιB
    inst✝¹ : DirectSum.GSemiring A
    inst✝ : DirectSum.Gmodule A M
    a b : DirectSum ιA fun i => A i
    c : DirectSum ιB fun i => M i
    ai : ιA
    ax : A ai
    bi : ιA
    bx : A bi
    ci : ιB
    cx : M ci
    ⊢ Eq ((((((((AddMonoidHom.compHom (DirectSum.Gmodule.smulAddMonoidHom A M)).co …
  -/
  dsimp only [coe_comp, Function.comp_apply, compHom_apply_apply, flip_apply, flipHom_apply]
  rw [smulAddMonoidHom_apply_of_of, smulAddMonoidHom_apply_of_of, DirectSum.mulHom_of_of,
    smulAddMonoidHom_apply_of_of]
  exact
    DirectSum.of_eq_of_gradedMonoid_eq
      (mul_smul (GradedMonoid.mk ai ax) (GradedMonoid.mk bi bx) (GradedMonoid.mk ci cx))


/-- The `Module` derived from `gmodule A M`. -/
instance module [DecidableEq ιA] [DecidableEq ιB] [GSemiring A] [Gmodule A M] :
    Module (⨁ i, A i) (⨁ i, M i) where
  smul := (· • ·)
  one_smul := one_smul' _ _
  mul_smul := mul_smul' _ _
  smul_add r := (smulAddMonoidHom A M r).map_add
  smul_zero r := (smulAddMonoidHom A M r).map_zero
                       /-
                         ιA : Type u_1
                         ιB : Type u_2
                         A : ιA → Type u_3
                         M : ιB → Type u_4
                         inst✝⁷ : AddMonoid ιA
                         inst✝⁶ : VAdd ιA ιB
                         inst✝⁵ : (i : ιA) → AddCommMonoid (A i)
                         inst✝⁴ : (i : ιB) → AddCommMonoid (M i)
                         inst✝³ : DecidableEq ιA
                         inst✝² : DecidableEq ιB
                         inst✝¹ : DirectSum.GSemiring A
                         inst✝ : DirectSum.Gmodule A M
                         r s : DirectSum ιA fun i => A i
                         x : DirectSum ιB fun i => M i
                         ⊢ Eq (HSMul.hSMul (HAdd.hAdd r s) x) (HAdd.hAdd (HSMul.hSMul r x) (HSMul.hSMul …
                       -/
  add_smul r s x := by simp only [smul_def, map_add, AddMonoidHom.add_apply]
                       /-
                         🎉 no goals
                       -/
                    /-
                      ιA : Type u_1
                      ιB : Type u_2
                      A : ιA → Type u_3
                      M : ιB → Type u_4
                      inst✝⁷ : AddMonoid ιA
                      inst✝⁶ : VAdd ιA ιB
                      inst✝⁵ : (i : ιA) → AddCommMonoid (A i)
                      inst✝⁴ : (i : ιB) → AddCommMonoid (M i)
                      inst✝³ : DecidableEq ιA
                      inst✝² : DecidableEq ιB
                      inst✝¹ : DirectSum.GSemiring A
                      inst✝ : DirectSum.Gmodule A M
                      x : DirectSum ιB fun i => M i
                      ⊢ Eq (HSMul.hSMul 0 x) 0
                    -/
  zero_smul x := by simp only [smul_def, map_zero, AddMonoidHom.zero_apply]
                    /-
                      🎉 no goals
                    -/


instance gmulAction [AddMonoid M] [DistribMulAction A M] [SetLike σ M] [SetLike.GradedMonoid 𝓐]
    [SetLike.GradedSMul 𝓐 𝓜] : GradedMonoid.GMulAction (fun i => 𝓐 i) fun i => 𝓜 i :=
  { SetLike.toGSMul 𝓐 𝓜 with
    one_smul := fun ⟨_i, _m⟩ => Sigma.subtype_ext (zero_vadd _ _) (one_smul _ _)
    mul_smul := fun ⟨_i, _a⟩ ⟨_j, _a'⟩ ⟨_k, _b⟩ =>
      Sigma.subtype_ext (add_vadd _ _ _) (mul_smul _ _ _) }


instance gdistribMulAction [AddMonoid M] [DistribMulAction A M] [SetLike σ M]
    [AddSubmonoidClass σ M] [SetLike.GradedMonoid 𝓐] [SetLike.GradedSMul 𝓐 𝓜] :
    DirectSum.GdistribMulAction (fun i => 𝓐 i) fun i => 𝓜 i :=
  { SetLike.gmulAction 𝓐 𝓜 with
    smul_add := fun _a _b _c => Subtype.ext <| smul_add _ _ _
    smul_zero := fun _a => Subtype.ext <| smul_zero _ }


/-- `[SetLike.GradedMonoid 𝓐] [SetLike.GradedSMul 𝓐 𝓜]` is the internal version of graded
  module, the internal version can be translated into the external version `gmodule`. -/
instance gmodule : DirectSum.Gmodule (fun i => 𝓐 i) fun i => 𝓜 i :=
  { SetLike.gdistribMulAction 𝓐 𝓜 with
    smul := fun x y => ⟨(x : A) • (y : M), SetLike.GradedSMul.smul_mem x.2 y.2⟩
    add_smul := fun _a _a' _b => Subtype.ext <| add_smul _ _ _
    zero_smul := fun _b => Subtype.ext <| zero_smul _ _ }


/-- The smul multiplication of `A` on `⨁ i, 𝓜 i` from `(⨁ i, 𝓐 i) →+ (⨁ i, 𝓜 i) →+ ⨁ i, 𝓜 i`
turns `⨁ i, 𝓜 i` into an `A`-module
-/
def isModule [DecidableEq ιA] [DecidableEq ιM] [GradedRing 𝓐] : Module A (⨁ i, 𝓜 i) :=
  { Module.compHom _ (DirectSum.decomposeRingEquiv 𝓐 : A ≃+* ⨁ i, 𝓐 i).toRingHom with
    smul := fun a b => DirectSum.decompose 𝓐 a • b }


/-- `⨁ i, 𝓜 i` and `M` are isomorphic as `A`-modules.
"The internal version" and "the external version" are isomorphism as `A`-modules.
-/
def linearEquiv [DecidableEq ιA] [DecidableEq ιM] [GradedRing 𝓐] [DirectSum.Decomposition 𝓜] :
    @LinearEquiv A A _ _ (RingHom.id A) (RingHom.id A) _ _ M (⨁ i, 𝓜 i) _
            /-
              ιA : Type u_1
              ιM : Type u_2
              R : Type u_3
              A : Type u_4
              M : Type u_5
              σ : Type u_6
              σ' : Type u_7
              inst✝¹⁶ : AddMonoid ιA
              inst✝¹⁵ : AddAction ιA ιM
              inst✝¹⁴ : CommSemiring R
              inst✝¹³ : Semiring A
              inst✝¹² : Algebra R A
              𝓐 : ιA → σ'
              inst✝¹¹ : SetLike σ' A
              𝓜 : ιM → σ
              inst✝¹⁰ : AddCommMonoid M
              inst✝⁹ : Module A M
              inst✝⁸ : SetLike σ M
              inst✝⁷ : AddSubmonoidClass σ' A
              inst✝⁶ : AddSubmonoidClass σ M
              inst✝⁵ : SetLike.GradedMonoid 𝓐
              inst✝⁴ : SetLike.GradedSMul 𝓐 𝓜
              inst✝³ : DecidableEq ιA
              inst✝² : DecidableEq ιM
              inst✝¹ : GradedRing 𝓐
              inst✝ : DirectSum.Decomposition 𝓜
              ⊢ Module A (DirectSum ιM fun i => Subtype fun x => Membership.mem (𝓜 i) x)
            -/
    _ _ (by letI := isModule 𝓐 𝓜; infer_instance) := by
                                  /-
                                    🎉 no goals
                                  -/
  /-
    ιA : Type u_1
    ιM : Type u_2
    R : Type u_3
    A : Type u_4
    M : Type u_5
    σ : Type u_6
    σ' : Type u_7
    inst✝¹⁶ : AddMonoid ιA
    inst✝¹⁵ : AddAction ιA ιM
    inst✝¹⁴ : CommSemiring R
    inst✝¹³ : Semiring A
    inst✝¹² : Algebra R A
    𝓐 : ιA → σ'
    inst✝¹¹ : SetLike σ' A
    𝓜 : ιM → σ
    inst✝¹⁰ : AddCommMonoid M
    inst✝⁹ : Module A M
    inst✝⁸ : SetLike σ M
    inst✝⁷ : AddSubmonoidClass σ' A
    inst✝⁶ : AddSubmonoidClass σ M
    inst✝⁵ : SetLike.GradedMonoid 𝓐
    inst✝⁴ : SetLike.GradedSMul 𝓐 𝓜
    inst✝³ : DecidableEq ιA
    inst✝² : DecidableEq ιM
    inst✝¹ : GradedRing 𝓐
    inst✝ : DirectSum.Decomposition 𝓜
    ⊢ LinearEquiv (RingHom.id A) M (DirectSum ιM fun i => Subtype fun x => Members …
  -/
  letI h := isModule 𝓐 𝓜
  refine ⟨⟨(DirectSum.decomposeAddEquiv 𝓜).toAddHom, ?_⟩,
    (DirectSum.decomposeAddEquiv 𝓜).symm.toFun, (DirectSum.decomposeAddEquiv 𝓜).left_inv,
    (DirectSum.decomposeAddEquiv 𝓜).right_inv⟩
  /-
    ιA : Type u_1
    ιM : Type u_2
    R : Type u_3
    A : Type u_4
    M : Type u_5
    σ : Type u_6
    σ' : Type u_7
    inst✝¹⁶ : AddMonoid ιA
    inst✝¹⁵ : AddAction ιA ιM
    inst✝¹⁴ : CommSemiring R
    inst✝¹³ : Semiring A
    inst✝¹² : Algebra R A
    𝓐 : ιA → σ'
    inst✝¹¹ : SetLike σ' A
    𝓜 : ιM → σ
    inst✝¹⁰ : AddCommMonoid M
    inst✝⁹ : Module A M
    inst✝⁸ : SetLike σ M
    inst✝⁷ : AddSubmonoidClass σ' A
    inst✝⁶ : AddSubmonoidClass σ M
    inst✝⁵ : SetLike.GradedMonoid 𝓐
    inst✝⁴ : SetLike.GradedSMul 𝓐 𝓜
    inst✝³ : DecidableEq ιA
    inst✝² : DecidableEq ιM
    inst✝¹ : GradedRing 𝓐
    inst✝ : DirectSum.Decomposition 𝓜
    h : Module A (DirectSum ιM fun i => Subtype fun x => Membership.mem (𝓜 i) x) : …
    ⊢ ∀ (m : A) (x : M), Eq ((DirectSum.decomposeAddEquiv 𝓜).toAddHom.toFun (HSMul …
  -/
  intro x y
  classical
  rw [AddHom.toFun_eq_coe, ← DirectSum.sum_support_decompose 𝓐 x, map_sum, Finset.sum_smul,
    AddEquiv.coe_toAddHom, map_sum, Finset.sum_smul]
  refine Finset.sum_congr rfl (fun i _hi => ?_)
  rw [RingHom.id_apply, ← DirectSum.sum_support_decompose 𝓜 y, map_sum, Finset.smul_sum, map_sum,
    Finset.smul_sum]
  refine Finset.sum_congr rfl (fun j _hj => ?_)
  rw [show (decompose 𝓐 x i : A) • (decomposeAddEquiv 𝓜 ↑(decompose 𝓜 y j) : (⨁ i, 𝓜 i)) =
    DirectSum.Gmodule.smulAddMonoidHom _ _ (decompose 𝓐 ↑(decompose 𝓐 x i))
    (decomposeAddEquiv 𝓜 ↑(decompose 𝓜 y j)) from DirectSum.Gmodule.smul_def _ _ _ _]
  simp only [decomposeAddEquiv_apply, Equiv.invFun_as_coe, Equiv.symm_symm, decompose_coe,
    Gmodule.smulAddMonoidHom_apply_of_of]
  convert DirectSum.decompose_coe 𝓜 _
  rfl


