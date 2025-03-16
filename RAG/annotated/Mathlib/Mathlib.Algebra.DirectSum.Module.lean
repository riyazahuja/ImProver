instance : Module R (⨁ i, M i) :=
  DFinsupp.module


instance {S : Type*} [Semiring S] [∀ i, Module S (M i)] [∀ i, SMulCommClass R S (M i)] :
    SMulCommClass R S (⨁ i, M i) :=
  DFinsupp.smulCommClass


instance {S : Type*} [Semiring S] [SMul R S] [∀ i, Module S (M i)] [∀ i, IsScalarTower R S (M i)] :
    IsScalarTower R S (⨁ i, M i) :=
  DFinsupp.isScalarTower


instance [∀ i, Module Rᵐᵒᵖ (M i)] [∀ i, IsCentralScalar R (M i)] : IsCentralScalar R (⨁ i, M i) :=
  DFinsupp.isCentralScalar


theorem smul_apply (b : R) (v : ⨁ i, M i) (i : ι) : (b • v) i = b • v i :=
  DFinsupp.smul_apply _ _ _


/-- Create the direct sum given a family `M` of `R` modules indexed over `ι`. -/
def lmk : ∀ s : Finset ι, (∀ i : (↑s : Set ι), M i.val) →ₗ[R] ⨁ i, M i :=
  DFinsupp.lmk


/-- Inclusion of each component into the direct sum. -/
def lof : ∀ i : ι, M i →ₗ[R] ⨁ i, M i :=
  DFinsupp.lsingle


theorem lof_eq_of (i : ι) (b : M i) : lof R ι M i b = of M i b := rfl


theorem single_eq_lof (i : ι) (b : M i) : DFinsupp.single i b = lof R ι M i b := rfl


/-- Scalar multiplication commutes with direct sums. -/
theorem mk_smul (s : Finset ι) (c : R) (x) : mk M s (c • x) = c • mk M s x :=
  (lmk R ι M s).map_smul c x


/-- Scalar multiplication commutes with the inclusion of each component into the direct sum. -/
theorem of_smul (i : ι) (c : R) (x) : of M i (c • x) = c • of M i x :=
  (lof R ι M i).map_smul c x


theorem support_smul [∀ (i : ι) (x : M i), Decidable (x ≠ 0)] (c : R) (v : ⨁ i, M i) :
    (c • v).support ⊆ v.support :=
  DFinsupp.support_smul _ _


/-- The linear map constructed using the universal property of the coproduct. -/
def toModule : (⨁ i, M i) →ₗ[R] N :=
  DFunLike.coe (DFinsupp.lsum ℕ) φ


/-- Coproducts in the categories of modules and additive monoids commute with the forgetful functor
from modules to additive monoids. -/
theorem coe_toModule_eq_coe_toAddMonoid :
    (toModule R ι N φ : (⨁ i, M i) → N) = toAddMonoid fun i ↦ (φ i).toAddMonoidHom := rfl


/-- The map constructed using the universal property gives back the original maps when
restricted to each component. -/
@[simp]
theorem toModule_lof (i) (x : M i) : toModule R ι N φ (lof R ι M i x) = φ i x :=
  toAddMonoid_of (fun i ↦ (φ i).toAddMonoidHom) i x


/-- Every linear map from a direct sum agrees with the one obtained by applying
the universal property to each of its components. -/
theorem toModule.unique (f : ⨁ i, M i) : ψ f = toModule R ι N (fun i ↦ ψ.comp <| lof R ι M i) f :=
  toAddMonoid.unique ψ.toAddMonoidHom f


/-- Two `LinearMap`s out of a direct sum are equal if they agree on the generators.

See note [partially-applied ext lemmas]. -/
@[ext]
theorem linearMap_ext ⦃ψ ψ' : (⨁ i, M i) →ₗ[R] N⦄
    (H : ∀ i, ψ.comp (lof R ι M i) = ψ'.comp (lof R ι M i)) : ψ = ψ' :=
  DFinsupp.lhom_ext' H


/-- The inclusion of a subset of the direct summands
into a larger subset of the direct summands, as a linear map. -/
def lsetToSet (S T : Set ι) (H : S ⊆ T) : (⨁ i : S, M i) →ₗ[R] ⨁ i : T, M i :=
  toModule R _ _ fun i ↦ lof R T (fun i : Subtype T ↦ M i) ⟨i, H i.prop⟩


/-- Given `Fintype α`, `linearEquivFunOnFintype R` is the natural `R`-linear equivalence
between `⨁ i, M i` and `∀ i, M i`. -/
@[simps apply]
def linearEquivFunOnFintype [Fintype ι] : (⨁ i, M i) ≃ₗ[R] ∀ i, M i :=
  { DFinsupp.equivFunOnFintype with
    toFun := (↑)
    map_add' := fun f g ↦ by
      /-
        R : Type u
        inst✝⁶ : Semiring R
        ι : Type v
        M : ι → Type w
        inst✝⁵ : (i : ι) → AddCommMonoid (M i)
        inst✝⁴ : (i : ι) → Module R (M i)
        inst✝³ : DecidableEq ι
        N : Type u₁
        inst✝² : AddCommMonoid N
        inst✝¹ : Module R N
        φ : (i : ι) → LinearMap (RingHom.id R) (M i) N
        ψ ψ' : LinearMap (RingHom.id R) (DirectSum ι fun i => M i) N
        inst✝ : Fintype ι
        f g : DirectSum ι fun i => M i
        ⊢ Eq (⇑(HAdd.hAdd f g)) (HAdd.hAdd ⇑f ⇑g)
      -/
      ext
      /-
        case h
        R : Type u
        inst✝⁶ : Semiring R
        ι : Type v
        M : ι → Type w
        inst✝⁵ : (i : ι) → AddCommMonoid (M i)
        inst✝⁴ : (i : ι) → Module R (M i)
        inst✝³ : DecidableEq ι
        N : Type u₁
        inst✝² : AddCommMonoid N
        inst✝¹ : Module R N
        φ : (i : ι) → LinearMap (RingHom.id R) (M i) N
        ψ ψ' : LinearMap (RingHom.id R) (DirectSum ι fun i => M i) N
        inst✝ : Fintype ι
        f g : DirectSum ι fun i => M i
        x✝ : ι
        ⊢ Eq ((HAdd.hAdd f g) x✝) (HAdd.hAdd (⇑f) (⇑g) x✝)
      -/
      rw [add_apply, Pi.add_apply]
      /-
        🎉 no goals
      -/
    map_smul' := fun c f ↦ by
      /-
        R : Type u
        inst✝⁶ : Semiring R
        ι : Type v
        M : ι → Type w
        inst✝⁵ : (i : ι) → AddCommMonoid (M i)
        inst✝⁴ : (i : ι) → Module R (M i)
        inst✝³ : DecidableEq ι
        N : Type u₁
        inst✝² : AddCommMonoid N
        inst✝¹ : Module R N
        φ : (i : ι) → LinearMap (RingHom.id R) (M i) N
        ψ ψ' : LinearMap (RingHom.id R) (DirectSum ι fun i => M i) N
        inst✝ : Fintype ι
        c : R
        f : DirectSum ι fun i => M i
        ⊢ Eq ({ toFun := DFunLike.coe, map_add' := ⋯ }.toFun (HSMul.hSMul c f)) (HSMul …
      -/
      simp_rw [RingHom.id_apply]
      /-
        R : Type u
        inst✝⁶ : Semiring R
        ι : Type v
        M : ι → Type w
        inst✝⁵ : (i : ι) → AddCommMonoid (M i)
        inst✝⁴ : (i : ι) → Module R (M i)
        inst✝³ : DecidableEq ι
        N : Type u₁
        inst✝² : AddCommMonoid N
        inst✝¹ : Module R N
        φ : (i : ι) → LinearMap (RingHom.id R) (M i) N
        ψ ψ' : LinearMap (RingHom.id R) (DirectSum ι fun i => M i) N
        inst✝ : Fintype ι
        c : R
        f : DirectSum ι fun i => M i
        ⊢ Eq (⇑(HSMul.hSMul c f)) (HSMul.hSMul c ⇑f)
      -/
      rw [DFinsupp.coe_smul] }
      /-
        🎉 no goals
      -/


@[simp]
theorem linearEquivFunOnFintype_lof [Fintype ι] (i : ι) (m : M i) :
    (linearEquivFunOnFintype R ι M) (lof R ι M i m) = Pi.single i m := by
  /-
    R : Type u
    inst✝⁴ : Semiring R
    ι : Type v
    M : ι → Type w
    inst✝³ : (i : ι) → AddCommMonoid (M i)
    inst✝² : (i : ι) → Module R (M i)
    inst✝¹ : DecidableEq ι
    inst✝ : Fintype ι
    i : ι
    m : M i
    ⊢ Eq ((DirectSum.linearEquivFunOnFintype R ι M) ((DirectSum.lof R ι M i) m)) ( …
  -/
  ext a
  /-
    case h
    R : Type u
    inst✝⁴ : Semiring R
    ι : Type v
    M : ι → Type w
    inst✝³ : (i : ι) → AddCommMonoid (M i)
    inst✝² : (i : ι) → Module R (M i)
    inst✝¹ : DecidableEq ι
    inst✝ : Fintype ι
    i : ι
    m : M i
    a : ι
    ⊢ Eq ((DirectSum.linearEquivFunOnFintype R ι M) ((DirectSum.lof R ι M i) m) a) …
  -/
  change (DFinsupp.equivFunOnFintype (lof R ι M i m)) a = _
  /-
    case h
    R : Type u
    inst✝⁴ : Semiring R
    ι : Type v
    M : ι → Type w
    inst✝³ : (i : ι) → AddCommMonoid (M i)
    inst✝² : (i : ι) → Module R (M i)
    inst✝¹ : DecidableEq ι
    inst✝ : Fintype ι
    i : ι
    m : M i
    a : ι
    ⊢ Eq (DFinsupp.equivFunOnFintype ((DirectSum.lof R ι M i) m) a) (Pi.single i m …
  -/
  convert _root_.congr_fun (DFinsupp.equivFunOnFintype_single i m) a
  /-
    🎉 no goals
  -/


@[simp]
theorem linearEquivFunOnFintype_symm_single [Fintype ι] (i : ι) (m : M i) :
    (linearEquivFunOnFintype R ι M).symm (Pi.single i m) = lof R ι M i m := by
  /-
    R : Type u
    inst✝⁴ : Semiring R
    ι : Type v
    M : ι → Type w
    inst✝³ : (i : ι) → AddCommMonoid (M i)
    inst✝² : (i : ι) → Module R (M i)
    inst✝¹ : DecidableEq ι
    inst✝ : Fintype ι
    i : ι
    m : M i
    ⊢ Eq ((DirectSum.linearEquivFunOnFintype R ι M).symm (Pi.single i m)) ((Direct …
  -/
  change (DFinsupp.equivFunOnFintype.symm (Pi.single i m)) = _
  /-
    R : Type u
    inst✝⁴ : Semiring R
    ι : Type v
    M : ι → Type w
    inst✝³ : (i : ι) → AddCommMonoid (M i)
    inst✝² : (i : ι) → Module R (M i)
    inst✝¹ : DecidableEq ι
    inst✝ : Fintype ι
    i : ι
    m : M i
    ⊢ Eq (DFinsupp.equivFunOnFintype.symm (Pi.single i m)) ((DirectSum.lof R ι M i …
  -/
  rw [DFinsupp.equivFunOnFintype_symm_single i m]
  /-
    R : Type u
    inst✝⁴ : Semiring R
    ι : Type v
    M : ι → Type w
    inst✝³ : (i : ι) → AddCommMonoid (M i)
    inst✝² : (i : ι) → Module R (M i)
    inst✝¹ : DecidableEq ι
    inst✝ : Fintype ι
    i : ι
    m : M i
    ⊢ Eq (DFinsupp.single i m) ((DirectSum.lof R ι M i) m)
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem linearEquivFunOnFintype_symm_coe [Fintype ι] (f : ⨁ i, M i) :
    (linearEquivFunOnFintype R ι M).symm f = f := by
  /-
    R : Type u
    inst✝³ : Semiring R
    ι : Type v
    M : ι → Type w
    inst✝² : (i : ι) → AddCommMonoid (M i)
    inst✝¹ : (i : ι) → Module R (M i)
    inst✝ : Fintype ι
    f : DirectSum ι fun i => M i
    ⊢ Eq ((DirectSum.linearEquivFunOnFintype R ι M).symm ⇑f) f
  -/
  simp [linearEquivFunOnFintype]
  /-
    🎉 no goals
  -/


/-- The natural linear equivalence between `⨁ _ : ι, M` and `M` when `Unique ι`. -/
protected def lid (M : Type v) (ι : Type* := PUnit) [AddCommMonoid M] [Module R M] [Unique ι] :
    (⨁ _ : ι, M) ≃ₗ[R] M :=
  { DirectSum.id M ι, toModule R ι M fun _ ↦ LinearMap.id with }


/-- The projection map onto one component, as a linear map. -/
def component (i : ι) : (⨁ i, M i) →ₗ[R] M i :=
  DFinsupp.lapply i


theorem apply_eq_component (f : ⨁ i, M i) (i : ι) : f i = component R ι M i f := rfl

-- Note(kmill): `@[ext]` cannot prove `ext_iff` because `R` is not determined by `f` or `g`.

@[ext (iff := false)]
theorem ext {f g : ⨁ i, M i} (h : ∀ i, component R ι M i f = component R ι M i g) : f = g :=
  DFinsupp.ext h


theorem ext_iff {f g : ⨁ i, M i} : f = g ↔ ∀ i, component R ι M i f = component R ι M i g :=
                /-
                  R : Type u
                  inst✝² : Semiring R
                  ι : Type v
                  M : ι → Type w
                  inst✝¹ : (i : ι) → AddCommMonoid (M i)
                  inst✝ : (i : ι) → Module R (M i)
                  f g : DirectSum ι fun i => M i
                  h : Eq f g
                  x✝ : ι
                  ⊢ Eq ((DirectSum.component R ι M x✝) f) ((DirectSum.component R ι M x✝) g)
                -/
  ⟨fun h _ ↦ by rw [h], ext R⟩
                /-
                  🎉 no goals
                -/


@[simp]
theorem lof_apply [DecidableEq ι] (i : ι) (b : M i) : ((lof R ι M i) b) i = b :=
  DFinsupp.single_eq_same


@[simp]
theorem component.lof_self [DecidableEq ι] (i : ι) (b : M i) :
    component R ι M i ((lof R ι M i) b) = b :=
  lof_apply R i b


theorem component.of [DecidableEq ι] (i j : ι) (b : M j) :
    component R ι M i ((lof R ι M j) b) = if h : j = i then Eq.recOn h b else 0 :=
  DFinsupp.single_apply


/-- The linear map between direct sums induced by a family of linear maps. -/
def lmap : (⨁ i, M i) →ₗ[R] ⨁ i, N i := DFinsupp.mapRange.linearMap f


@[simp] theorem lmap_apply (x i) : lmap f x i = f i (x i) := rfl


@[simp] theorem lmap_lof [DecidableEq ι] (i) (x : M i) :
    lmap f (lof R _ _ _ x) = lof R _ _ _ (f i x) :=
  DFinsupp.mapRange_single (hf := fun _ ↦ map_zero _)


theorem lmap_injective : Function.Injective (lmap f) ↔ ∀ i, Function.Injective (f i) := by
  /-
    R : Type u
    inst✝⁴ : Semiring R
    ι : Type v
    M : ι → Type w
    inst✝³ : (i : ι) → AddCommMonoid (M i)
    inst✝² : (i : ι) → Module R (M i)
    N : ι → Type u_1
    inst✝¹ : (i : ι) → AddCommMonoid (N i)
    inst✝ : (i : ι) → Module R (N i)
    f : (i : ι) → LinearMap (RingHom.id R) (M i) (N i)
    ⊢ Iff (Function.Injective ⇑(DirectSum.lmap f)) (∀ (i : ι), Function.Injective  …
  -/
  classical exact DFinsupp.mapRange_injective (hf := fun _ ↦ map_zero _)
  /-
    🎉 no goals
  -/


/-- Reindexing terms of a direct sum is linear. -/
def lequivCongrLeft (h : ι ≃ κ) : (⨁ i, M i) ≃ₗ[R] ⨁ k, M (h.symm k) :=
  { equivCongrLeft h with map_smul' := DFinsupp.comapDomain'_smul h.invFun h.right_inv }


@[simp]
theorem lequivCongrLeft_apply (h : ι ≃ κ) (f : ⨁ i, M i) (k : κ) :
    lequivCongrLeft R h f k = f (h.symm k) :=
  equivCongrLeft_apply _ _ _


/-- `curry` as a linear map. -/
def sigmaLcurry : (⨁ i : Σ_, _, δ i.1 i.2) →ₗ[R] ⨁ (i) (j), δ i j :=
                                            /-
                                              R : Type u
                                              inst✝⁵ : Semiring R
                                              ι : Type v
                                              M : ι → Type w
                                              inst✝⁴ : (i : ι) → AddCommMonoid (M i)
                                              inst✝³ : (i : ι) → Module R (M i)
                                              α : ι → Type u_1
                                              δ : (i : ι) → α i → Type w
                                              inst✝² : DecidableEq ι
                                              inst✝¹ : (i : ι) → (j : α i) → AddCommMonoid (δ i j)
                                              inst✝ : (i : ι) → (j : α i) → Module R (δ i j)
                                              r : R
                                              ⊢ ∀ (x : DirectSum (Sigma fun x => α x) fun i => δ i.fst i.snd), Eq ({ toFun : …
                                            -/
  { sigmaCurry with map_smul' := fun r ↦ by convert DFinsupp.sigmaCurry_smul (δ := δ) r }
                                            /-
                                              🎉 no goals
                                            -/


@[simp]
theorem sigmaLcurry_apply (f : ⨁ i : Σ_, _, δ i.1 i.2) (i : ι) (j : α i) :
    sigmaLcurry R f i j = f ⟨i, j⟩ :=
  sigmaCurry_apply f i j


/-- `uncurry` as a linear map. -/
def sigmaLuncurry : (⨁ (i) (j), δ i j) →ₗ[R] ⨁ i : Σ_, _, δ i.1 i.2 :=
  { sigmaUncurry with map_smul' := DFinsupp.sigmaUncurry_smul }


@[simp]
theorem sigmaLuncurry_apply (f : ⨁ (i) (j), δ i j) (i : ι) (j : α i) :
    sigmaLuncurry R f ⟨i, j⟩ = f i j :=
  sigmaUncurry_apply f i j


/-- `curryEquiv` as a linear equiv. -/
def sigmaLcurryEquiv : (⨁ i : Σ_, _, δ i.1 i.2) ≃ₗ[R] ⨁ (i) (j), δ i j :=
  { sigmaCurryEquiv, sigmaLcurry R with }


/-- Linear isomorphism obtained by separating the term of index `none` of a direct sum over
`Option ι`. -/
@[simps]
noncomputable def lequivProdDirectSum : (⨁ i, α i) ≃ₗ[R] α none × ⨁ i, α (some i) :=
  { addEquivProdDirectSum with map_smul' := DFinsupp.equivProdDFinsupp_smul }


/-- The canonical linear map from `⨁ i, A i` to `M` where `A` is a collection of `Submodule R M`
indexed by `ι`. This is `DirectSum.coeAddMonoidHom` as a `LinearMap`. -/
def coeLinearMap : (⨁ i, A i) →ₗ[R] M :=
  toModule R ι M fun i ↦ (A i).subtype


theorem coeLinearMap_eq_dfinsupp_sum [DecidableEq M] (x : DirectSum ι fun i => A i) :
    coeLinearMap A x = DFinsupp.sum x fun i => (fun x : A i => ↑x) := by
  simp only [coeLinearMap, toModule, DFinsupp.lsum, LinearEquiv.coe_mk, LinearMap.coe_mk,
    AddHom.coe_mk]
  /-
    R : Type u
    inst✝³ : Semiring R
    ι : Type v
    dec_ι : DecidableEq ι
    M : Type u_1
    inst✝² : AddCommMonoid M
    inst✝¹ : Module R M
    A : ι → Submodule R M
    inst✝ : DecidableEq M
    x : DirectSum ι fun i => Subtype fun x => Membership.mem (A i) x
    ⊢ Eq ((DFinsupp.sumAddHom fun i => (A i).subtype.toAddMonoidHom) x) (DFinsupp. …
  -/
  rw [DFinsupp.sumAddHom_apply]
  /-
    R : Type u
    inst✝³ : Semiring R
    ι : Type v
    dec_ι : DecidableEq ι
    M : Type u_1
    inst✝² : AddCommMonoid M
    inst✝¹ : Module R M
    A : ι → Submodule R M
    inst✝ : DecidableEq M
    x : DirectSum ι fun i => Subtype fun x => Membership.mem (A i) x
    ⊢ Eq (DFinsupp.sum x fun x => ⇑(A x).subtype.toAddMonoidHom) (DFinsupp.sum x f …
  -/
  simp only [LinearMap.toAddMonoidHom_coe, Submodule.coe_subtype]
  /-
    🎉 no goals
  -/


@[simp]
theorem coeLinearMap_of (i : ι) (x : A i) : DirectSum.coeLinearMap A (of (fun i ↦ A i) i x) = x :=
  -- Porting note: spelled out arguments. (I don't know how this works.)
  toAddMonoid_of (β := fun i => A i) (fun i ↦ ((A i).subtype : A i →+ M)) i x


theorem range_coeLinearMap : LinearMap.range (coeLinearMap A) = ⨆ i, A i :=
  (Submodule.iSup_eq_range_dfinsupp_lsum _).symm


@[simp]
theorem IsInternal.ofBijective_coeLinearMap_same (h : IsInternal A)
    {i : ι} (x : A i) :
    (LinearEquiv.ofBijective (coeLinearMap A) h).symm x i = x := by
  /-
    R : Type u
    inst✝² : Semiring R
    ι : Type v
    dec_ι : DecidableEq ι
    M : Type u_1
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    A : ι → Submodule R M
    h : DirectSum.IsInternal A
    i : ι
    x : Subtype fun x => Membership.mem (A i) x
    ⊢ Eq (((LinearEquiv.ofBijective (DirectSum.coeLinearMap A) h).symm ↑x) i) x
  -/
  rw [← coeLinearMap_of, LinearEquiv.ofBijective_symm_apply_apply, of_eq_same]
  /-
    🎉 no goals
  -/


@[simp]
theorem IsInternal.ofBijective_coeLinearMap_of_ne (h : IsInternal A)
    {i j : ι} (hij : i ≠ j) (x : A i) :
    (LinearEquiv.ofBijective (coeLinearMap A) h).symm x j = 0 := by
  /-
    R : Type u
    inst✝² : Semiring R
    ι : Type v
    dec_ι : DecidableEq ι
    M : Type u_1
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    A : ι → Submodule R M
    h : DirectSum.IsInternal A
    i j : ι
    hij : Ne i j
    x : Subtype fun x => Membership.mem (A i) x
    ⊢ Eq (((LinearEquiv.ofBijective (DirectSum.coeLinearMap A) h).symm ↑x) j) 0
  -/
  rw [← coeLinearMap_of, LinearEquiv.ofBijective_symm_apply_apply, of_eq_of_ne i j _ hij]
  /-
    🎉 no goals
  -/


theorem IsInternal.ofBijective_coeLinearMap_of_mem (h : IsInternal A)
    {i : ι} {x : M} (hx : x ∈ A i) :
    (LinearEquiv.ofBijective (coeLinearMap A) h).symm x i = ⟨x, hx⟩ :=
  h.ofBijective_coeLinearMap_same ⟨x, hx⟩


theorem IsInternal.ofBijective_coeLinearMap_of_mem_ne (h : IsInternal A)
    {i j : ι} (hij : i ≠ j) {x : M} (hx : x ∈ A i) :
    (LinearEquiv.ofBijective (coeLinearMap A) h).symm x j = 0 :=
  h.ofBijective_coeLinearMap_of_ne hij ⟨x, hx⟩


/-- If a direct sum of submodules is internal then the submodules span the module. -/
theorem IsInternal.submodule_iSup_eq_top (h : IsInternal A) : iSup A = ⊤ := by
  /-
    R : Type u
    inst✝² : Semiring R
    ι : Type v
    dec_ι : DecidableEq ι
    M : Type u_1
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    A : ι → Submodule R M
    h : DirectSum.IsInternal A
    ⊢ Eq (iSup A) Top.top
  -/
  rw [Submodule.iSup_eq_range_dfinsupp_lsum, LinearMap.range_eq_top]
  /-
    R : Type u
    inst✝² : Semiring R
    ι : Type v
    dec_ι : DecidableEq ι
    M : Type u_1
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    A : ι → Submodule R M
    h : DirectSum.IsInternal A
    ⊢ Function.Surjective ⇑((DFinsupp.lsum Nat) fun i => (A i).subtype)
  -/
  exact Function.Bijective.surjective h
  /-
    🎉 no goals
  -/


/-- If a direct sum of submodules is internal then the submodules are independent. -/
theorem IsInternal.submodule_iSupIndep (h : IsInternal A) : iSupIndep A :=
  iSupIndep_of_dfinsupp_lsum_injective _ h.injective


@[deprecated (since := "2024-11-24")]
alias IsInternal.submodule_independent := IsInternal.submodule_iSupIndep


/-- Given an internal direct sum decomposition of a module `M`, and a basis for each of the
components of the direct sum, the disjoint union of these bases is a basis for `M`. -/
noncomputable def IsInternal.collectedBasis (h : IsInternal A) {α : ι → Type*}
    (v : ∀ i, Basis (α i) R (A i)) : Basis (Σi, α i) R M where
  repr :=
    ((LinearEquiv.ofBijective (DirectSum.coeLinearMap A) h).symm ≪≫ₗ
        DFinsupp.mapRange.linearEquiv fun i ↦ (v i).repr) ≪≫ₗ
      (sigmaFinsuppLequivDFinsupp R).symm


@[simp]
theorem IsInternal.collectedBasis_coe (h : IsInternal A) {α : ι → Type*}
    (v : ∀ i, Basis (α i) R (A i)) : ⇑(h.collectedBasis v) = fun a : Σi, α i ↦ ↑(v a.1 a.2) := by
  /-
    R : Type u
    inst✝² : Semiring R
    ι : Type v
    dec_ι : DecidableEq ι
    M : Type u_1
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    A : ι → Submodule R M
    h : DirectSum.IsInternal A
    α : ι → Type u_2
    v : (i : ι) → Basis (α i) R (Subtype fun x => Membership.mem (A i) x)
    ⊢ Eq ⇑(h.collectedBasis v) fun a => ↑((v a.fst) a.snd)
  -/
  funext a
  -- Porting note: was
  -- simp only [IsInternal.collectedBasis, toModule, coeLinearMap, Basis.coe_ofRepr,
  --   Basis.repr_symm_apply, DFinsupp.lsum_apply_apply, DFinsupp.mapRange.linearEquiv_apply,
  --   DFinsupp.mapRange.linearEquiv_symm, DFinsupp.mapRange_single, linearCombination_single,
  --   LinearEquiv.ofBijective_apply, LinearEquiv.symm_symm, LinearEquiv.symm_trans_apply, one_smul,
  --   sigmaFinsuppAddEquivDFinsupp_apply, sigmaFinsuppEquivDFinsupp_single,
  --   sigmaFinsuppLequivDFinsupp_apply]
  -- convert DFinsupp.sumAddHom_single (fun i ↦ (A i).subtype.toAddMonoidHom) a.1 (v a.1 a.2)
  simp only [IsInternal.collectedBasis, coeLinearMap, Basis.coe_ofRepr, LinearEquiv.trans_symm,
    LinearEquiv.symm_symm, LinearEquiv.trans_apply, sigmaFinsuppLequivDFinsupp_apply,
    sigmaFinsuppEquivDFinsupp_single, LinearEquiv.ofBijective_apply,
    sigmaFinsuppAddEquivDFinsupp_apply]
  /-
    case h
    R : Type u
    inst✝² : Semiring R
    ι : Type v
    dec_ι : DecidableEq ι
    M : Type u_1
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    A : ι → Submodule R M
    h : DirectSum.IsInternal A
    α : ι → Type u_2
    v : (i : ι) → Basis (α i) R (Subtype fun x => Membership.mem (A i) x)
    a : Sigma fun i => α i
    ⊢ Eq ((DirectSum.toModule R ι M fun i => (A i).subtype) ((DFinsupp.mapRange.li …
  -/
  rw [DFinsupp.mapRange.linearEquiv_symm]
  /-
    case h
    R : Type u
    inst✝² : Semiring R
    ι : Type v
    dec_ι : DecidableEq ι
    M : Type u_1
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    A : ι → Submodule R M
    h : DirectSum.IsInternal A
    α : ι → Type u_2
    v : (i : ι) → Basis (α i) R (Subtype fun x => Membership.mem (A i) x)
    a : Sigma fun i => α i
    ⊢ Eq ((DirectSum.toModule R ι M fun i => (A i).subtype) ((DFinsupp.mapRange.li …
  -/
  erw [DFinsupp.mapRange.linearEquiv_apply]
  simp only [DFinsupp.mapRange_single, Basis.repr_symm_apply, linearCombination_single, one_smul,
    toModule]
  /-
    case h
    R : Type u
    inst✝² : Semiring R
    ι : Type v
    dec_ι : DecidableEq ι
    M : Type u_1
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    A : ι → Submodule R M
    h : DirectSum.IsInternal A
    α : ι → Type u_2
    v : (i : ι) → Basis (α i) R (Subtype fun x => Membership.mem (A i) x)
    a : Sigma fun i => α i
    ⊢ Eq (((DFinsupp.lsum Nat) fun i => (A i).subtype) (DFinsupp.single a.fst ((v  …
  -/
  erw [DFinsupp.lsum_single]
  /-
    case h
    R : Type u
    inst✝² : Semiring R
    ι : Type v
    dec_ι : DecidableEq ι
    M : Type u_1
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    A : ι → Submodule R M
    h : DirectSum.IsInternal A
    α : ι → Type u_2
    v : (i : ι) → Basis (α i) R (Subtype fun x => Membership.mem (A i) x)
    a : Sigma fun i => α i
    ⊢ Eq ((A a.fst).subtype ((v a.fst) a.snd)) ↑((v a.fst) a.snd)
  -/
  simp only [Submodule.coe_subtype]
  /-
    🎉 no goals
  -/


theorem IsInternal.collectedBasis_mem (h : IsInternal A) {α : ι → Type*}
                                                                                      /-
                                                                                        R : Type u
                                                                                        inst✝² : Semiring R
                                                                                        ι : Type v
                                                                                        dec_ι : DecidableEq ι
                                                                                        M : Type u_1
                                                                                        inst✝¹ : AddCommMonoid M
                                                                                        inst✝ : Module R M
                                                                                        A : ι → Submodule R M
                                                                                        h : DirectSum.IsInternal A
                                                                                        α : ι → Type u_2
                                                                                        v : (i : ι) → Basis (α i) R (Subtype fun x => Membership.mem (A i) x)
                                                                                        a : Sigma fun i => α i
                                                                                        ⊢ Membership.mem (A a.fst) ((h.collectedBasis v) a)
                                                                                      -/
    (v : ∀ i, Basis (α i) R (A i)) (a : Σi, α i) : h.collectedBasis v a ∈ A a.1 := by simp
                                                                                      /-
                                                                                        🎉 no goals
                                                                                      -/


theorem IsInternal.collectedBasis_repr_of_mem (h : IsInternal A) {α : ι → Type*}
    (v : ∀ i, Basis (α i) R (A i)) {x : M} {i : ι} {a : α i} (hx : x ∈ A i) :
    (h.collectedBasis v).repr x ⟨i, a⟩ = (v i).repr ⟨x, hx⟩ a := by
  /-
    R : Type u
    inst✝² : Semiring R
    ι : Type v
    dec_ι : DecidableEq ι
    M : Type u_1
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    A : ι → Submodule R M
    h : DirectSum.IsInternal A
    α : ι → Type u_2
    v : (i : ι) → Basis (α i) R (Subtype fun x => Membership.mem (A i) x)
    x : M
    i : ι
    a : α i
    hx : Membership.mem (A i) x
    ⊢ Eq (((h.collectedBasis v).repr x) ⟨i, a⟩) (((v i).repr ⟨x, hx⟩) a)
  -/
  change (sigmaFinsuppLequivDFinsupp R).symm (DFinsupp.mapRange _ (fun i ↦ map_zero _) _) _ = _
  /-
    R : Type u
    inst✝² : Semiring R
    ι : Type v
    dec_ι : DecidableEq ι
    M : Type u_1
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    A : ι → Submodule R M
    h : DirectSum.IsInternal A
    α : ι → Type u_2
    v : (i : ι) → Basis (α i) R (Subtype fun x => Membership.mem (A i) x)
    x : M
    i : ι
    a : α i
    hx : Membership.mem (A i) x
    ⊢ Eq (((sigmaFinsuppLequivDFinsupp R).symm (DFinsupp.mapRange (fun i => ⇑(v i) …
  -/
  simp [h.ofBijective_coeLinearMap_of_mem hx]
  /-
    🎉 no goals
  -/


theorem IsInternal.collectedBasis_repr_of_mem_ne (h : IsInternal A) {α : ι → Type*}
    (v : ∀ i, Basis (α i) R (A i)) {x : M} {i j : ι} (hij : i ≠ j) {a : α j} (hx : x ∈ A i) :
    (h.collectedBasis v).repr x ⟨j, a⟩ = 0 := by
  /-
    R : Type u
    inst✝² : Semiring R
    ι : Type v
    dec_ι : DecidableEq ι
    M : Type u_1
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    A : ι → Submodule R M
    h : DirectSum.IsInternal A
    α : ι → Type u_2
    v : (i : ι) → Basis (α i) R (Subtype fun x => Membership.mem (A i) x)
    x : M
    i j : ι
    hij : Ne i j
    a : α j
    hx : Membership.mem (A i) x
    ⊢ Eq (((h.collectedBasis v).repr x) ⟨j, a⟩) 0
  -/
  change (sigmaFinsuppLequivDFinsupp R).symm (DFinsupp.mapRange _ (fun i ↦ map_zero _) _) _ = _
  /-
    R : Type u
    inst✝² : Semiring R
    ι : Type v
    dec_ι : DecidableEq ι
    M : Type u_1
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    A : ι → Submodule R M
    h : DirectSum.IsInternal A
    α : ι → Type u_2
    v : (i : ι) → Basis (α i) R (Subtype fun x => Membership.mem (A i) x)
    x : M
    i j : ι
    hij : Ne i j
    a : α j
    hx : Membership.mem (A i) x
    ⊢ Eq (((sigmaFinsuppLequivDFinsupp R).symm (DFinsupp.mapRange (fun i => ⇑(v i) …
  -/
  simp [h.ofBijective_coeLinearMap_of_mem_ne hij hx]
  /-
    🎉 no goals
  -/


/-- When indexed by only two distinct elements, `DirectSum.IsInternal` implies
the two submodules are complementary. Over a `Ring R`, this is true as an iff, as
`DirectSum.isInternal_submodule_iff_isCompl`. -/
theorem IsInternal.isCompl {A : ι → Submodule R M} {i j : ι} (hij : i ≠ j)
    (h : (Set.univ : Set ι) = {i, j}) (hi : IsInternal A) : IsCompl (A i) (A j) :=
  ⟨hi.submodule_iSupIndep.pairwiseDisjoint hij,
    codisjoint_iff.mpr <| Eq.symm <| hi.submodule_iSup_eq_top.symm.trans <| by
      /-
        R : Type u
        inst✝² : Semiring R
        ι : Type v
        dec_ι : DecidableEq ι
        M : Type u_1
        inst✝¹ : AddCommMonoid M
        inst✝ : Module R M
        A : ι → Submodule R M
        i j : ι
        hij : Ne i j
        h : Eq Set.univ (Insert.insert i (Singleton.singleton j))
        hi : DirectSum.IsInternal A
        ⊢ Eq (iSup A) (Max.max (A i) (A j))
      -/
      rw [← sSup_pair, iSup, ← Set.image_univ, h, Set.image_insert_eq, Set.image_singleton]⟩
      /-
        🎉 no goals
      -/


/-- Note that this is not generally true for `[Semiring R]`; see
`iSupIndep.dfinsupp_lsum_injective` for details. -/
theorem isInternal_submodule_of_iSupIndep_of_iSup_eq_top {A : ι → Submodule R M}
    (hi : iSupIndep A) (hs : iSup A = ⊤) : IsInternal A :=
  ⟨hi.dfinsupp_lsum_injective,
    -- Note: https://github.com/leanprover-community/mathlib4/pull/8386 had to specify value of `f`
    (LinearMap.range_eq_top (f := DFinsupp.lsum _ _)).1 <|
      (Submodule.iSup_eq_range_dfinsupp_lsum _).symm.trans hs⟩


@[deprecated (since := "2024-11-24")]
alias isInternal_submodule_of_independent_of_iSup_eq_top :=
  isInternal_submodule_of_iSupIndep_of_iSup_eq_top


/-- `iff` version of `DirectSum.isInternal_submodule_of_iSupIndep_of_iSup_eq_top`,
`DirectSum.IsInternal.iSupIndep`, and `DirectSum.IsInternal.submodule_iSup_eq_top`. -/
theorem isInternal_submodule_iff_iSupIndep_and_iSup_eq_top (A : ι → Submodule R M) :
    IsInternal A ↔ iSupIndep A ∧ iSup A = ⊤ :=
  ⟨fun i ↦ ⟨i.submodule_iSupIndep, i.submodule_iSup_eq_top⟩,
    And.rec isInternal_submodule_of_iSupIndep_of_iSup_eq_top⟩


@[deprecated (since := "2024-11-24")]
alias isInternal_submodule_iff_independent_and_iSup_eq_top :=
  isInternal_submodule_iff_iSupIndep_and_iSup_eq_top


/-- If a collection of submodules has just two indices, `i` and `j`, then
`DirectSum.IsInternal` is equivalent to `isCompl`. -/
theorem isInternal_submodule_iff_isCompl (A : ι → Submodule R M) {i j : ι} (hij : i ≠ j)
    (h : (Set.univ : Set ι) = {i, j}) : IsInternal A ↔ IsCompl (A i) (A j) := by
  /-
    R : Type u
    inst✝² : Ring R
    ι : Type v
    dec_ι : DecidableEq ι
    M : Type u_1
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    A : ι → Submodule R M
    i j : ι
    hij : Ne i j
    h : Eq Set.univ (Insert.insert i (Singleton.singleton j))
    ⊢ Iff (DirectSum.IsInternal A) (IsCompl (A i) (A j))
  -/
  have : ∀ k, k = i ∨ k = j := fun k ↦ by simpa using Set.ext_iff.mp h k
  rw [isInternal_submodule_iff_iSupIndep_and_iSup_eq_top, iSup, ← Set.image_univ, h,
    Set.image_insert_eq, Set.image_singleton, sSup_pair, iSupIndep_pair hij this]
  /-
    R : Type u
    inst✝² : Ring R
    ι : Type v
    dec_ι : DecidableEq ι
    M : Type u_1
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    A : ι → Submodule R M
    i j : ι
    hij : Ne i j
    h : Eq Set.univ (Insert.insert i (Singleton.singleton j))
    this : ∀ (k : ι), Or (Eq k i) (Eq k j)
    ⊢ Iff (And (Disjoint (A i) (A j)) (Eq (Max.max (A i) (A j)) Top.top)) (IsCompl …
  -/
  exact ⟨fun ⟨hd, ht⟩ ↦ ⟨hd, codisjoint_iff.mpr ht⟩, fun ⟨hd, ht⟩ ↦ ⟨hd, ht.eq_top⟩⟩
  /-
    🎉 no goals
  -/


@[simp]
theorem isInternal_ne_bot_iff {A : ι → Submodule R M} :
    IsInternal (fun i : {i // A i ≠ ⊥} ↦ A i) ↔ IsInternal A := by
  /-
    R : Type u
    inst✝² : Ring R
    ι : Type v
    dec_ι : DecidableEq ι
    M : Type u_1
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    A : ι → Submodule R M
    ⊢ Iff (DirectSum.IsInternal fun i => A ↑i) (DirectSum.IsInternal A)
  -/
  simp [isInternal_submodule_iff_iSupIndep_and_iSup_eq_top]
  /-
    🎉 no goals
  -/


lemma isInternal_biSup_submodule_of_iSupIndep {A : ι → Submodule R M} (s : Set ι)
    (h : iSupIndep <| fun i : s ↦ A i) :
    IsInternal <| fun (i : s) ↦ (A i).comap (⨆ i ∈ s, A i).subtype := by
  /-
    R : Type u
    inst✝² : Ring R
    ι : Type v
    dec_ι : DecidableEq ι
    M : Type u_1
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    A : ι → Submodule R M
    s : Set ι
    h : iSupIndep fun i => A ↑i
    ⊢ DirectSum.IsInternal fun i => Submodule.comap (iSup fun i => iSup fun h => A …
  -/
  refine (isInternal_submodule_iff_iSupIndep_and_iSup_eq_top _).mpr ⟨?_, by simp [iSup_subtype]⟩
  /-
    R : Type u
    inst✝² : Ring R
    ι : Type v
    dec_ι : DecidableEq ι
    M : Type u_1
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    A : ι → Submodule R M
    s : Set ι
    h : iSupIndep fun i => A ↑i
    ⊢ iSupIndep fun i => Submodule.comap (iSup fun i => iSup fun h => A i).subtype …
  -/
  let p := ⨆ i ∈ s, A i
  /-
    R : Type u
    inst✝² : Ring R
    ι : Type v
    dec_ι : DecidableEq ι
    M : Type u_1
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    A : ι → Submodule R M
    s : Set ι
    h : iSupIndep fun i => A ↑i
    p : Submodule R M := iSup fun i => iSup fun h => A i
    ⊢ iSupIndep fun i => Submodule.comap (iSup fun i => iSup fun h => A i).subtype …
  -/
  have hp : ∀ i ∈ s, A i ≤ p := fun i hi ↦ le_biSup A hi
  /-
    R : Type u
    inst✝² : Ring R
    ι : Type v
    dec_ι : DecidableEq ι
    M : Type u_1
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    A : ι → Submodule R M
    s : Set ι
    h : iSupIndep fun i => A ↑i
    p : Submodule R M := iSup fun i => iSup fun h => A i
    hp : ∀ (i : ι), Membership.mem s i → LE.le (A i) p
    ⊢ iSupIndep fun i => Submodule.comap (iSup fun i => iSup fun h => A i).subtype …
  -/
  let e : Submodule R p ≃o Set.Iic p := p.mapIic
  suffices (e ∘ fun i : s ↦ (A i).comap p.subtype) = fun i ↦ ⟨A i, hp i i.property⟩ by
    rw [← iSupIndep_map_orderIso_iff e, this]
    exact .of_coe_Iic_comp h
  /-
    R : Type u
    inst✝² : Ring R
    ι : Type v
    dec_ι : DecidableEq ι
    M : Type u_1
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    A : ι → Submodule R M
    s : Set ι
    h : iSupIndep fun i => A ↑i
    p : Submodule R M := iSup fun i => iSup fun h => A i
    hp : ∀ (i : ι), Membership.mem s i → LE.le (A i) p
    e : OrderIso (Submodule R (Subtype fun x => Membership.mem p x)) ↑(Set.Iic p)  …
    ⊢ Eq (Function.comp ⇑e fun i => Submodule.comap p.subtype (A ↑i)) fun i => ⟨A  …
  -/
  ext i m
  /-
    case h.a.h
    R : Type u
    inst✝² : Ring R
    ι : Type v
    dec_ι : DecidableEq ι
    M : Type u_1
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    A : ι → Submodule R M
    s : Set ι
    h : iSupIndep fun i => A ↑i
    p : Submodule R M := iSup fun i => iSup fun h => A i
    hp : ∀ (i : ι), Membership.mem s i → LE.le (A i) p
    e : OrderIso (Submodule R (Subtype fun x => Membership.mem p x)) ↑(Set.Iic p)  …
    i : ↑s
    m : M
    ⊢ Iff (Membership.mem (↑(Function.comp (⇑e) (fun i => Submodule.comap p.subtyp …
  -/
  change m ∈ ((A i).comap p.subtype).map p.subtype ↔ _
  /-
    case h.a.h
    R : Type u
    inst✝² : Ring R
    ι : Type v
    dec_ι : DecidableEq ι
    M : Type u_1
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    A : ι → Submodule R M
    s : Set ι
    h : iSupIndep fun i => A ↑i
    p : Submodule R M := iSup fun i => iSup fun h => A i
    hp : ∀ (i : ι), Membership.mem s i → LE.le (A i) p
    e : OrderIso (Submodule R (Subtype fun x => Membership.mem p x)) ↑(Set.Iic p)  …
    i : ↑s
    m : M
    ⊢ Iff (Membership.mem (Submodule.map p.subtype (Submodule.comap p.subtype (A ↑ …
  -/
  rw [Submodule.map_comap_subtype, inf_of_le_right (hp i i.property)]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-11-24")]
alias isInternal_biSup_submodule_of_independent := isInternal_biSup_submodule_of_iSupIndep


theorem IsInternal.addSubmonoid_iSupIndep {M : Type*} [AddCommMonoid M] {A : ι → AddSubmonoid M}
    (h : IsInternal A) : iSupIndep A :=
  iSupIndep_of_dfinsupp_sumAddHom_injective _ h.injective


@[deprecated (since := "2024-11-24")]
alias IsInternal.addSubmonoid_independent := IsInternal.addSubmonoid_iSupIndep


theorem IsInternal.addSubgroup_iSupIndep {G : Type*} [AddCommGroup G] {A : ι → AddSubgroup G}
    (h : IsInternal A) : iSupIndep A :=
  iSupIndep_of_dfinsupp_sumAddHom_injective' _ h.injective


@[deprecated (since := "2024-11-24")]
alias IsInternal.addSubgroup_independent := IsInternal.addSubgroup_iSupIndep


