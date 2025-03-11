/-- `DFinsupp.mk` as a `LinearMap`. -/
def lmk (s : Finset ι) : (∀ i : (↑s : Set ι), M i) →ₗ[R] Π₀ i, M i where
  toFun := mk s
  map_add' _ _ := mk_add
  map_smul' c x := mk_smul c x


/-- `DFinsupp.single` as a `LinearMap` -/
def lsingle (i) : M i →ₗ[R] Π₀ i, M i :=
  { DFinsupp.singleAddHom _ _ with
    toFun := single i
    map_smul' := single_smul }


/-- Two `R`-linear maps from `Π₀ i, M i` which agree on each `single i x` agree everywhere. -/
theorem lhom_ext ⦃φ ψ : (Π₀ i, M i) →ₗ[R] N⦄ (h : ∀ i x, φ (single i x) = ψ (single i x)) : φ = ψ :=
  LinearMap.toAddMonoidHom_injective <| addHom_ext h


/-- Two `R`-linear maps from `Π₀ i, M i` which agree on each `single i x` agree everywhere.

See note [partially-applied ext lemmas].
After applying this lemma, if `M = R` then it suffices to verify
`φ (single a 1) = ψ (single a 1)`. -/
@[ext 1100]
theorem lhom_ext' ⦃φ ψ : (Π₀ i, M i) →ₗ[R] N⦄ (h : ∀ i, φ.comp (lsingle i) = ψ.comp (lsingle i)) :
    φ = ψ :=
  lhom_ext fun i => LinearMap.congr_fun (h i)

-- This lemma has always been bad, but the linter only noticed after https://github.com/leanprover/lean4/pull/2644.

@[simp, nolint simpNF]
theorem lmk_apply (s : Finset ι) (x) : (lmk s : _ →ₗ[R] Π₀ i, M i) x = mk s x :=
  rfl


@[simp]
theorem lsingle_apply (i : ι) (x : M i) : (lsingle i : (M i) →ₗ[R] _) x = single i x :=
  rfl


/-- Interpret `fun (f : Π₀ i, M i) ↦ f i` as a linear map. -/
def lapply (i : ι) : (Π₀ i, M i) →ₗ[R] M i where
  toFun f := f i
  map_add' f g := add_apply f g i
  map_smul' c f := smul_apply c f i


@[simp]
theorem lapply_apply (i : ι) (f : Π₀ i, M i) : (lapply i : (Π₀ i, M i) →ₗ[R] _) f = f i :=
  rfl


@[simp]
theorem lapply_comp_lsingle_same [DecidableEq ι] (i : ι) :
                                                        /-
                                                          ι : Type u_1
                                                          R : Type u_2
                                                          M : ι → Type u_4
                                                          inst✝³ : Semiring R
                                                          inst✝² : (i : ι) → AddCommMonoid (M i)
                                                          inst✝¹ : (i : ι) → Module R (M i)
                                                          inst✝ : DecidableEq ι
                                                          i : ι
                                                          ⊢ Eq ((DFinsupp.lapply i).comp (DFinsupp.lsingle i)) LinearMap.id
                                                        -/
    lapply i ∘ₗ lsingle i = (.id : M i →ₗ[R] M i) := by ext; simp
                                                             /-
                                                               🎉 no goals
                                                             -/


@[simp]
theorem lapply_comp_lsingle_of_ne [DecidableEq ι] (i i' : ι) (h : i ≠ i') :
                                                        /-
                                                          ι : Type u_1
                                                          R : Type u_2
                                                          M : ι → Type u_4
                                                          inst✝³ : Semiring R
                                                          inst✝² : (i : ι) → AddCommMonoid (M i)
                                                          inst✝¹ : (i : ι) → Module R (M i)
                                                          inst✝ : DecidableEq ι
                                                          i i' : ι
                                                          h : Ne i i'
                                                          ⊢ Eq ((DFinsupp.lapply i).comp (DFinsupp.lsingle i')) 0
                                                        -/
    lapply i ∘ₗ lsingle i' = (0 : M i' →ₗ[R] M i) := by ext; simp [h.symm]
                                                             /-
                                                               🎉 no goals
                                                             -/


/-- Typeclass inference can't find `DFinsupp.addCommMonoid` without help for this case.
This instance allows it to be found where it is needed on the LHS of the colon in
`DFinsupp.moduleOfLinearMap`. -/
instance addCommMonoidOfLinearMap : AddCommMonoid (Π₀ i : ι, M i →ₗ[R] N) :=
  inferInstance


/-- Typeclass inference can't find `DFinsupp.module` without help for this case.
This is needed to define `DFinsupp.lsum` below.

The cause seems to be an inability to unify the `∀ i, AddCommMonoid (M i →ₗ[R] N)` instance that
we have with the `∀ i, Zero (M i →ₗ[R] N)` instance which appears as a parameter to the
`DFinsupp` type. -/
instance moduleOfLinearMap [Semiring S] [Module S N] [SMulCommClass R S N] :
    Module S (Π₀ i : ι, M i →ₗ[R] N) :=
  DFinsupp.module


instance {R : Type*} {S : Type*} [Semiring R] [Semiring S] (σ : R →+* S)
    {σ' : S →+* R} [RingHomInvPair σ σ'] [RingHomInvPair σ' σ] (M : Type*) (M₂ : Type*)
    [AddCommMonoid M] [AddCommMonoid M₂] [Module R M] [Module S M₂] :
    EquivLike (LinearEquiv σ M M₂) M M₂ :=
  inferInstance

/- Porting note: In every application of lsum that follows, the argument M needs to be explicitly
supplied, lean does not manage to gather that information itself -/

/-- The `DFinsupp` version of `Finsupp.lsum`.

See note [bundled maps over different rings] for why separate `R` and `S` semirings are used. -/
@[simps]
def lsum [Semiring S] [Module S N] [SMulCommClass R S N] :
    (∀ i, M i →ₗ[R] N) ≃ₗ[S] (Π₀ i, M i) →ₗ[R] N where
  toFun F :=
    { toFun := sumAddHom fun i => (F i).toAddMonoidHom
      map_add' := (DFinsupp.liftAddHom fun (i : ι) => (F i).toAddMonoidHom).map_add
      map_smul' := fun c f => by
        /-
          ι : Type u_1
          R : Type u_2
          S : Type u_3
          M : ι → Type u_4
          N : Type u_5
          inst✝⁸ : Semiring R
          inst✝⁷ : (i : ι) → AddCommMonoid (M i)
          inst✝⁶ : (i : ι) → Module R (M i)
          inst✝⁵ : AddCommMonoid N
          inst✝⁴ : Module R N
          inst✝³ : DecidableEq ι
          inst✝² : Semiring S
          inst✝¹ : Module S N
          inst✝ : SMulCommClass R S N
          F : (i : ι) → LinearMap (RingHom.id R) (M i) N
          c : R
          f : DFinsupp fun i => M i
          ⊢ Eq ({ toFun := ⇑(DFinsupp.sumAddHom fun i => (F i).toAddMonoidHom), map_add' …
        -/
        dsimp
        /-
          ι : Type u_1
          R : Type u_2
          S : Type u_3
          M : ι → Type u_4
          N : Type u_5
          inst✝⁸ : Semiring R
          inst✝⁷ : (i : ι) → AddCommMonoid (M i)
          inst✝⁶ : (i : ι) → Module R (M i)
          inst✝⁵ : AddCommMonoid N
          inst✝⁴ : Module R N
          inst✝³ : DecidableEq ι
          inst✝² : Semiring S
          inst✝¹ : Module S N
          inst✝ : SMulCommClass R S N
          F : (i : ι) → LinearMap (RingHom.id R) (M i) N
          c : R
          f : DFinsupp fun i => M i
          ⊢ Eq ((DFinsupp.sumAddHom fun i => (F i).toAddMonoidHom) (HSMul.hSMul c f)) (H …
        -/
        apply DFinsupp.induction f
          /-
            case h0
            ι : Type u_1
            R : Type u_2
            S : Type u_3
            M : ι → Type u_4
            N : Type u_5
            inst✝⁸ : Semiring R
            inst✝⁷ : (i : ι) → AddCommMonoid (M i)
            inst✝⁶ : (i : ι) → Module R (M i)
            inst✝⁵ : AddCommMonoid N
            inst✝⁴ : Module R N
            inst✝³ : DecidableEq ι
            inst✝² : Semiring S
            inst✝¹ : Module S N
            inst✝ : SMulCommClass R S N
            F : (i : ι) → LinearMap (RingHom.id R) (M i) N
            c : R
            f : DFinsupp fun i => M i
            ⊢ Eq ((DFinsupp.sumAddHom fun i => (F i).toAddMonoidHom) (HSMul.hSMul c 0)) (H …
          -/
        · rw [smul_zero, AddMonoidHom.map_zero, smul_zero]
          /-
            🎉 no goals
          -/
          /-
            case ha
            ι : Type u_1
            R : Type u_2
            S : Type u_3
            M : ι → Type u_4
            N : Type u_5
            inst✝⁸ : Semiring R
            inst✝⁷ : (i : ι) → AddCommMonoid (M i)
            inst✝⁶ : (i : ι) → Module R (M i)
            inst✝⁵ : AddCommMonoid N
            inst✝⁴ : Module R N
            inst✝³ : DecidableEq ι
            inst✝² : Semiring S
            inst✝¹ : Module S N
            inst✝ : SMulCommClass R S N
            F : (i : ι) → LinearMap (RingHom.id R) (M i) N
            c : R
            f : DFinsupp fun i => M i
            ⊢ ∀ (i : ι) (b : M i) (f : DFinsupp fun i => M i), Eq (f i) 0 → Ne b 0 → Eq (( …
          -/
        · intro a b f _ _ hf
          rw [smul_add, AddMonoidHom.map_add, AddMonoidHom.map_add, smul_add, hf, ← single_smul,
            sumAddHom_single, sumAddHom_single, LinearMap.toAddMonoidHom_coe,
            LinearMap.map_smul] }
  invFun F i := F.comp (lsingle i)
  left_inv F := by
    /-
      ι : Type u_1
      R : Type u_2
      S : Type u_3
      M : ι → Type u_4
      N : Type u_5
      inst✝⁸ : Semiring R
      inst✝⁷ : (i : ι) → AddCommMonoid (M i)
      inst✝⁶ : (i : ι) → Module R (M i)
      inst✝⁵ : AddCommMonoid N
      inst✝⁴ : Module R N
      inst✝³ : DecidableEq ι
      inst✝² : Semiring S
      inst✝¹ : Module S N
      inst✝ : SMulCommClass R S N
      F : (i : ι) → LinearMap (RingHom.id R) (M i) N
      ⊢ Eq ((fun F i => F.comp (DFinsupp.lsingle i)) ({ toFun := fun F => { toFun := …
    -/
    ext
    /-
      case h.h
      ι : Type u_1
      R : Type u_2
      S : Type u_3
      M : ι → Type u_4
      N : Type u_5
      inst✝⁸ : Semiring R
      inst✝⁷ : (i : ι) → AddCommMonoid (M i)
      inst✝⁶ : (i : ι) → Module R (M i)
      inst✝⁵ : AddCommMonoid N
      inst✝⁴ : Module R N
      inst✝³ : DecidableEq ι
      inst✝² : Semiring S
      inst✝¹ : Module S N
      inst✝ : SMulCommClass R S N
      F : (i : ι) → LinearMap (RingHom.id R) (M i) N
      x✝¹ : ι
      x✝ : M x✝¹
      ⊢ Eq (((fun F i => F.comp (DFinsupp.lsingle i)) ({ toFun := fun F => { toFun : …
    -/
    simp
    /-
      🎉 no goals
    -/
  right_inv F := by
    /-
      ι : Type u_1
      R : Type u_2
      S : Type u_3
      M : ι → Type u_4
      N : Type u_5
      inst✝⁸ : Semiring R
      inst✝⁷ : (i : ι) → AddCommMonoid (M i)
      inst✝⁶ : (i : ι) → Module R (M i)
      inst✝⁵ : AddCommMonoid N
      inst✝⁴ : Module R N
      inst✝³ : DecidableEq ι
      inst✝² : Semiring S
      inst✝¹ : Module S N
      inst✝ : SMulCommClass R S N
      F : LinearMap (RingHom.id R) (DFinsupp fun i => M i) N
      ⊢ Eq ({ toFun := fun F => { toFun := ⇑(DFinsupp.sumAddHom fun i => (F i).toAdd …
    -/
    /-
      ι : Type u_1
      R : Type u_2
      S : Type u_3
      M : ι → Type u_4
      N : Type u_5
      inst✝⁸ : Semiring R
      inst✝⁷ : (i : ι) → AddCommMonoid (M i)
      inst✝⁶ : (i : ι) → Module R (M i)
      inst✝⁵ : AddCommMonoid N
      inst✝⁴ : Module R N
      inst✝³ : DecidableEq ι
      inst✝² : Semiring S
      inst✝¹ : Module S N
      inst✝ : SMulCommClass R S N
      F G : (i : ι) → LinearMap (RingHom.id R) (M i) N
      ⊢ Eq ((fun F => { toFun := ⇑(DFinsupp.sumAddHom fun i => (F i).toAddMonoidHom) …
    -/
    refine DFinsupp.lhom_ext' (fun i ↦ ?_)
    /-
      ι : Type u_1
      R : Type u_2
      S : Type u_3
      M : ι → Type u_4
      N : Type u_5
      inst✝⁸ : Semiring R
      inst✝⁷ : (i : ι) → AddCommMonoid (M i)
      inst✝⁶ : (i : ι) → Module R (M i)
      inst✝⁵ : AddCommMonoid N
      inst✝⁴ : Module R N
      inst✝³ : DecidableEq ι
      inst✝² : Semiring S
      inst✝¹ : Module S N
      inst✝ : SMulCommClass R S N
      F G : (i : ι) → LinearMap (RingHom.id R) (M i) N
      i : ι
      ⊢ Eq (((fun F => { toFun := ⇑(DFinsupp.sumAddHom fun i => (F i).toAddMonoidHom …
    -/
    /-
      ι : Type u_1
      R : Type u_2
      S : Type u_3
      M : ι → Type u_4
      N : Type u_5
      inst✝⁸ : Semiring R
      inst✝⁷ : (i : ι) → AddCommMonoid (M i)
      inst✝⁶ : (i : ι) → Module R (M i)
      inst✝⁵ : AddCommMonoid N
      inst✝⁴ : Module R N
      inst✝³ : DecidableEq ι
      inst✝² : Semiring S
      inst✝¹ : Module S N
      inst✝ : SMulCommClass R S N
      F : LinearMap (RingHom.id R) (DFinsupp fun i => M i) N
      i : ι
      ⊢ Eq (({ toFun := fun F => { toFun := ⇑(DFinsupp.sumAddHom fun i => (F i).toAd …
    -/
    /-
      case h
      ι : Type u_1
      R : Type u_2
      S : Type u_3
      M : ι → Type u_4
      N : Type u_5
      inst✝⁸ : Semiring R
      inst✝⁷ : (i : ι) → AddCommMonoid (M i)
      inst✝⁶ : (i : ι) → Module R (M i)
      inst✝⁵ : AddCommMonoid N
      inst✝⁴ : Module R N
      inst✝³ : DecidableEq ι
      inst✝² : Semiring S
      inst✝¹ : Module S N
      inst✝ : SMulCommClass R S N
      F G : (i : ι) → LinearMap (RingHom.id R) (M i) N
      i : ι
      x✝ : M i
      ⊢ Eq ((((fun F => { toFun := ⇑(DFinsupp.sumAddHom fun i => (F i).toAddMonoidHo …
    -/
    ext
    /-
      🎉 no goals
    -/
    /-
      case h
      ι : Type u_1
      R : Type u_2
      S : Type u_3
      M : ι → Type u_4
      N : Type u_5
      inst✝⁸ : Semiring R
      inst✝⁷ : (i : ι) → AddCommMonoid (M i)
      inst✝⁶ : (i : ι) → Module R (M i)
      inst✝⁵ : AddCommMonoid N
      inst✝⁴ : Module R N
      inst✝³ : DecidableEq ι
      inst✝² : Semiring S
      inst✝¹ : Module S N
      inst✝ : SMulCommClass R S N
      F : LinearMap (RingHom.id R) (DFinsupp fun i => M i) N
      i : ι
      x✝ : M i
      ⊢ Eq ((({ toFun := fun F => { toFun := ⇑(DFinsupp.sumAddHom fun i => (F i).toA …
    -/
    /-
      ι : Type u_1
      R : Type u_2
      S : Type u_3
      M : ι → Type u_4
      N : Type u_5
      inst✝⁸ : Semiring R
      inst✝⁷ : (i : ι) → AddCommMonoid (M i)
      inst✝⁶ : (i : ι) → Module R (M i)
      inst✝⁵ : AddCommMonoid N
      inst✝⁴ : Module R N
      inst✝³ : DecidableEq ι
      inst✝² : Semiring S
      inst✝¹ : Module S N
      inst✝ : SMulCommClass R S N
      c : S
      F : (i : ι) → LinearMap (RingHom.id R) (M i) N
      ⊢ Eq ({ toFun := fun F => { toFun := ⇑(DFinsupp.sumAddHom fun i => (F i).toAdd …
    -/
    simp
    /-
      ι : Type u_1
      R : Type u_2
      S : Type u_3
      M : ι → Type u_4
      N : Type u_5
      inst✝⁸ : Semiring R
      inst✝⁷ : (i : ι) → AddCommMonoid (M i)
      inst✝⁶ : (i : ι) → Module R (M i)
      inst✝⁵ : AddCommMonoid N
      inst✝⁴ : Module R N
      inst✝³ : DecidableEq ι
      inst✝² : Semiring S
      inst✝¹ : Module S N
      inst✝ : SMulCommClass R S N
      c : S
      F : (i : ι) → LinearMap (RingHom.id R) (M i) N
      i : ι
      ⊢ Eq (({ toFun := fun F => { toFun := ⇑(DFinsupp.sumAddHom fun i => (F i).toAd …
    -/
    /-
      🎉 no goals
    -/
    /-
      case h
      ι : Type u_1
      R : Type u_2
      S : Type u_3
      M : ι → Type u_4
      N : Type u_5
      inst✝⁸ : Semiring R
      inst✝⁷ : (i : ι) → AddCommMonoid (M i)
      inst✝⁶ : (i : ι) → Module R (M i)
      inst✝⁵ : AddCommMonoid N
      inst✝⁴ : Module R N
      inst✝³ : DecidableEq ι
      inst✝² : Semiring S
      inst✝¹ : Module S N
      inst✝ : SMulCommClass R S N
      c : S
      F : (i : ι) → LinearMap (RingHom.id R) (M i) N
      i : ι
      x✝ : M i
      ⊢ Eq ((({ toFun := fun F => { toFun := ⇑(DFinsupp.sumAddHom fun i => (F i).toA …
    -/
  map_add' F G := by
    /-
      🎉 no goals
    -/
    refine DFinsupp.lhom_ext' (fun i ↦ ?_)
    ext
    simp
  map_smul' c F := by
    refine DFinsupp.lhom_ext' (fun i ↦ ?_)
    ext
    simp


/-- While `simp` can prove this, it is often convenient to avoid unfolding `lsum` into `sumAddHom`
with `DFinsupp.lsum_apply_apply`. -/
theorem lsum_single [Semiring S] [Module S N] [SMulCommClass R S N] (F : ∀ i, M i →ₗ[R] N) (i)
    (x : M i) : lsum S (M := M) F (single i x) = F i x := by
  /-
    ι : Type u_1
    R : Type u_2
    S : Type u_3
    M : ι → Type u_4
    N : Type u_5
    inst✝⁸ : Semiring R
    inst✝⁷ : (i : ι) → AddCommMonoid (M i)
    inst✝⁶ : (i : ι) → Module R (M i)
    inst✝⁵ : AddCommMonoid N
    inst✝⁴ : Module R N
    inst✝³ : DecidableEq ι
    inst✝² : Semiring S
    inst✝¹ : Module S N
    inst✝ : SMulCommClass R S N
    F : (i : ι) → LinearMap (RingHom.id R) (M i) N
    i : ι
    x : M i
    ⊢ Eq (((DFinsupp.lsum S) F) (DFinsupp.single i x)) ((F i) x)
  -/
  simp
  /-
    🎉 no goals
  -/


theorem mapRange_smul (f : ∀ i, β₁ i → β₂ i) (hf : ∀ i, f i 0 = 0) (r : R)
    (hf' : ∀ i x, f i (r • x) = r • f i x) (g : Π₀ i, β₁ i) :
    mapRange f hf (r • g) = r • mapRange f hf g := by
  /-
    ι : Type u_1
    R : Type u_2
    inst✝⁴ : Semiring R
    β₁ : ι → Type u_7
    β₂ : ι → Type u_8
    inst✝³ : (i : ι) → AddCommMonoid (β₁ i)
    inst✝² : (i : ι) → AddCommMonoid (β₂ i)
    inst✝¹ : (i : ι) → Module R (β₁ i)
    inst✝ : (i : ι) → Module R (β₂ i)
    f : (i : ι) → β₁ i → β₂ i
    hf : ∀ (i : ι), Eq (f i 0) 0
    r : R
    hf' : ∀ (i : ι) (x : β₁ i), Eq (f i (HSMul.hSMul r x)) (HSMul.hSMul r (f i x))
    g : DFinsupp fun i => β₁ i
    ⊢ Eq (DFinsupp.mapRange f hf (HSMul.hSMul r g)) (HSMul.hSMul r (DFinsupp.mapRa …
  -/
  ext
  /-
    case h
    ι : Type u_1
    R : Type u_2
    inst✝⁴ : Semiring R
    β₁ : ι → Type u_7
    β₂ : ι → Type u_8
    inst✝³ : (i : ι) → AddCommMonoid (β₁ i)
    inst✝² : (i : ι) → AddCommMonoid (β₂ i)
    inst✝¹ : (i : ι) → Module R (β₁ i)
    inst✝ : (i : ι) → Module R (β₂ i)
    f : (i : ι) → β₁ i → β₂ i
    hf : ∀ (i : ι), Eq (f i 0) 0
    r : R
    hf' : ∀ (i : ι) (x : β₁ i), Eq (f i (HSMul.hSMul r x)) (HSMul.hSMul r (f i x))
    g : DFinsupp fun i => β₁ i
    i✝ : ι
    ⊢ Eq ((DFinsupp.mapRange f hf (HSMul.hSMul r g)) i✝) ((HSMul.hSMul r (DFinsupp …
  -/
  simp only [mapRange_apply f, coe_smul, Pi.smul_apply, hf']
  /-
    🎉 no goals
  -/


/-- `DFinsupp.mapRange` as a `LinearMap`. -/
@[simps! apply]
def mapRange.linearMap (f : ∀ i, β₁ i →ₗ[R] β₂ i) : (Π₀ i, β₁ i) →ₗ[R] Π₀ i, β₂ i :=
  { mapRange.addMonoidHom fun i => (f i).toAddMonoidHom with
    toFun := mapRange (fun i x => f i x) fun i => (f i).map_zero
    map_smul' := fun r => mapRange_smul _ (fun i => (f i).map_zero) _ fun i => (f i).map_smul r }


@[simp]
theorem mapRange.linearMap_id :
    (mapRange.linearMap fun i => (LinearMap.id : β₂ i →ₗ[R] _)) = LinearMap.id := by
  /-
    ι : Type u_1
    R : Type u_2
    inst✝² : Semiring R
    β₂ : ι → Type u_8
    inst✝¹ : (i : ι) → AddCommMonoid (β₂ i)
    inst✝ : (i : ι) → Module R (β₂ i)
    ⊢ Eq (DFinsupp.mapRange.linearMap fun i => LinearMap.id) LinearMap.id
  -/
  ext
  /-
    case h.h
    ι : Type u_1
    R : Type u_2
    inst✝² : Semiring R
    β₂ : ι → Type u_8
    inst✝¹ : (i : ι) → AddCommMonoid (β₂ i)
    inst✝ : (i : ι) → Module R (β₂ i)
    x✝ : DFinsupp fun i => β₂ i
    i✝ : ι
    ⊢ Eq (((DFinsupp.mapRange.linearMap fun i => LinearMap.id) x✝) i✝) ((LinearMap …
  -/
  simp [linearMap]
  /-
    🎉 no goals
  -/


theorem mapRange.linearMap_comp (f : ∀ i, β₁ i →ₗ[R] β₂ i) (f₂ : ∀ i, β i →ₗ[R] β₁ i) :
    (mapRange.linearMap fun i => (f i).comp (f₂ i)) =
      (mapRange.linearMap f).comp (mapRange.linearMap f₂) :=
  LinearMap.ext <| mapRange_comp (fun i x => f i x) (fun i x => f₂ i x)
                                                             /-
                                                               ι : Type u_1
                                                               R : Type u_2
                                                               inst✝⁶ : Semiring R
                                                               β : ι → Type u_6
                                                               β₁ : ι → Type u_7
                                                               β₂ : ι → Type u_8
                                                               inst✝⁵ : (i : ι) → AddCommMonoid (β i)
                                                               inst✝⁴ : (i : ι) → AddCommMonoid (β₁ i)
                                                               inst✝³ : (i : ι) → AddCommMonoid (β₂ i)
                                                               inst✝² : (i : ι) → Module R (β i)
                                                               inst✝¹ : (i : ι) → Module R (β₁ i)
                                                               inst✝ : (i : ι) → Module R (β₂ i)
                                                               f : (i : ι) → LinearMap (RingHom.id R) (β₁ i) (β₂ i)
                                                               f₂ : (i : ι) → LinearMap (RingHom.id R) (β i) (β₁ i)
                                                               ⊢ ∀ (i : ι), Eq (Function.comp ((fun i x => (f i) x) i) ((fun i x => (f₂ i) x) …
                                                             -/
    (fun i => (f i).map_zero) (fun i => (f₂ i).map_zero) (by simp)
                                                             /-
                                                               🎉 no goals
                                                             -/


theorem sum_mapRange_index.linearMap [DecidableEq ι] {f : ∀ i, β₁ i →ₗ[R] β₂ i}
    {h : ∀ i, β₂ i →ₗ[R] N} {l : Π₀ i, β₁ i} :
    DFinsupp.lsum ℕ h (mapRange.linearMap f l) = DFinsupp.lsum ℕ (fun i => (h i).comp (f i)) l := by
  /-
    ι : Type u_1
    R : Type u_2
    N : Type u_5
    inst✝⁷ : Semiring R
    inst✝⁶ : AddCommMonoid N
    inst✝⁵ : Module R N
    β₁ : ι → Type u_7
    β₂ : ι → Type u_8
    inst✝⁴ : (i : ι) → AddCommMonoid (β₁ i)
    inst✝³ : (i : ι) → AddCommMonoid (β₂ i)
    inst✝² : (i : ι) → Module R (β₁ i)
    inst✝¹ : (i : ι) → Module R (β₂ i)
    inst✝ : DecidableEq ι
    f : (i : ι) → LinearMap (RingHom.id R) (β₁ i) (β₂ i)
    h : (i : ι) → LinearMap (RingHom.id R) (β₂ i) N
    l : DFinsupp fun i => β₁ i
    ⊢ Eq (((DFinsupp.lsum Nat) h) ((DFinsupp.mapRange.linearMap f) l)) (((DFinsupp …
  -/
  classical simpa [DFinsupp.sumAddHom_apply] using sum_mapRange_index fun i => by simp
  /-
    🎉 no goals
  -/


/-- `DFinsupp.mapRange.linearMap` as a `LinearEquiv`. -/
@[simps apply]
def mapRange.linearEquiv (e : ∀ i, β₁ i ≃ₗ[R] β₂ i) : (Π₀ i, β₁ i) ≃ₗ[R] Π₀ i, β₂ i :=
  { mapRange.addEquiv fun i => (e i).toAddEquiv,
    mapRange.linearMap fun i => (e i).toLinearMap with
    toFun := mapRange (fun i x => e i x) fun i => (e i).map_zero
    invFun := mapRange (fun i x => (e i).symm x) fun i => (e i).symm.map_zero }


@[simp]
theorem mapRange.linearEquiv_refl :
    (mapRange.linearEquiv fun i => LinearEquiv.refl R (β₁ i)) = LinearEquiv.refl _ _ :=
  LinearEquiv.ext mapRange_id


theorem mapRange.linearEquiv_trans (f : ∀ i, β i ≃ₗ[R] β₁ i) (f₂ : ∀ i, β₁ i ≃ₗ[R] β₂ i) :
    (mapRange.linearEquiv fun i => (f i).trans (f₂ i)) =
      (mapRange.linearEquiv f).trans (mapRange.linearEquiv f₂) :=
  LinearEquiv.ext <| mapRange_comp (fun i x => f₂ i x) (fun i x => f i x)
                                                             /-
                                                               ι : Type u_1
                                                               R : Type u_2
                                                               inst✝⁶ : Semiring R
                                                               β : ι → Type u_6
                                                               β₁ : ι → Type u_7
                                                               β₂ : ι → Type u_8
                                                               inst✝⁵ : (i : ι) → AddCommMonoid (β i)
                                                               inst✝⁴ : (i : ι) → AddCommMonoid (β₁ i)
                                                               inst✝³ : (i : ι) → AddCommMonoid (β₂ i)
                                                               inst✝² : (i : ι) → Module R (β i)
                                                               inst✝¹ : (i : ι) → Module R (β₁ i)
                                                               inst✝ : (i : ι) → Module R (β₂ i)
                                                               f : (i : ι) → LinearEquiv (RingHom.id R) (β i) (β₁ i)
                                                               f₂ : (i : ι) → LinearEquiv (RingHom.id R) (β₁ i) (β₂ i)
                                                               ⊢ ∀ (i : ι), Eq (Function.comp ((fun i x => (f₂ i) x) i) ((fun i x => (f i) x) …
                                                             -/
    (fun i => (f₂ i).map_zero) (fun i => (f i).map_zero) (by simp)
                                                             /-
                                                               🎉 no goals
                                                             -/


@[simp]
theorem mapRange.linearEquiv_symm (e : ∀ i, β₁ i ≃ₗ[R] β₂ i) :
    (mapRange.linearEquiv e).symm = mapRange.linearEquiv fun i => (e i).symm :=
  rfl


/-- Given a family of linear maps `f i : M i →ₗ[R] N`, we can form a linear map
`(Π₀ i, M i) →ₗ[R] N` which sends `x : Π₀ i, M i` to the sum over `i` of `f i` applied to `x i`.
This is the map coming from the universal property of `Π₀ i, M i` as the coproduct of the `M i`.
See also `LinearMap.coprod` for the binary product version. -/
def coprodMap (f : ∀ i : ι, M i →ₗ[R] N) : (Π₀ i, M i) →ₗ[R] N :=
  (DFinsupp.lsum ℕ fun _ : ι => LinearMap.id) ∘ₗ DFinsupp.mapRange.linearMap f


theorem coprodMap_apply [∀ x : N, Decidable (x ≠ 0)] (f : ∀ i : ι, M i →ₗ[R] N) (x : Π₀ i, M i) :
    coprodMap f x =
      DFinsupp.sum (mapRange (fun i => f i) (fun _ => LinearMap.map_zero _) x) fun _ =>
        id :=
  DFinsupp.sumAddHom_apply _ _


theorem coprodMap_apply_single (f : ∀ i : ι, M i →ₗ[R] N) (i : ι) (x : M i) :
    coprodMap f (single i x) = f i x := by
  /-
    ι : Type u_1
    R : Type u_2
    M : ι → Type u_4
    N : Type u_5
    inst✝⁵ : Semiring R
    inst✝⁴ : (i : ι) → AddCommMonoid (M i)
    inst✝³ : (i : ι) → Module R (M i)
    inst✝² : AddCommMonoid N
    inst✝¹ : Module R N
    inst✝ : DecidableEq ι
    f : (i : ι) → LinearMap (RingHom.id R) (M i) N
    i : ι
    x : M i
    ⊢ Eq ((DFinsupp.coprodMap f) (DFinsupp.single i x)) ((f i) x)
  -/
  simp [coprodMap]
  /-
    🎉 no goals
  -/


theorem dfinsupp_sum_mem {β : ι → Type*} [∀ i, Zero (β i)] [∀ (i) (x : β i), Decidable (x ≠ 0)]
    (S : Submodule R N) (f : Π₀ i, β i) (g : ∀ i, β i → N) (h : ∀ c, f c ≠ 0 → g c (f c) ∈ S) :
    f.sum g ∈ S :=
  _root_.dfinsupp_sum_mem S f g h


theorem dfinsupp_sumAddHom_mem {β : ι → Type*} [∀ i, AddZeroClass (β i)] (S : Submodule R N)
    (f : Π₀ i, β i) (g : ∀ i, β i →+ N) (h : ∀ c, f c ≠ 0 → g c (f c) ∈ S) :
    DFinsupp.sumAddHom g f ∈ S :=
  _root_.dfinsupp_sumAddHom_mem S f g h


/-- The supremum of a family of submodules is equal to the range of `DFinsupp.lsum`; that is
every element in the `iSup` can be produced from taking a finite number of non-zero elements
of `p i`, coercing them to `N`, and summing them. -/
theorem iSup_eq_range_dfinsupp_lsum (p : ι → Submodule R N) :
    iSup p = LinearMap.range (DFinsupp.lsum ℕ (M := fun i ↦ ↥(p i)) fun i => (p i).subtype) := by
  /-
    ι : Type u_1
    R : Type u_2
    N : Type u_5
    inst✝³ : Semiring R
    inst✝² : AddCommMonoid N
    inst✝¹ : Module R N
    inst✝ : DecidableEq ι
    p : ι → Submodule R N
    ⊢ Eq (iSup p) (LinearMap.range ((DFinsupp.lsum Nat) fun i => (p i).subtype))
  -/
  apply le_antisymm
    /-
      case a
      ι : Type u_1
      R : Type u_2
      N : Type u_5
      inst✝³ : Semiring R
      inst✝² : AddCommMonoid N
      inst✝¹ : Module R N
      inst✝ : DecidableEq ι
      p : ι → Submodule R N
      ⊢ LE.le (iSup p) (LinearMap.range ((DFinsupp.lsum Nat) fun i => (p i).subtype))
    -/
  · apply iSup_le _
    /-
      ι : Type u_1
      R : Type u_2
      N : Type u_5
      inst✝³ : Semiring R
      inst✝² : AddCommMonoid N
      inst✝¹ : Module R N
      inst✝ : DecidableEq ι
      p : ι → Submodule R N
      ⊢ ∀ (i : ι), LE.le (p i) (LinearMap.range ((DFinsupp.lsum Nat) fun i => (p i). …
    -/
    intro i y hy
    /-
      ι : Type u_1
      R : Type u_2
      N : Type u_5
      inst✝³ : Semiring R
      inst✝² : AddCommMonoid N
      inst✝¹ : Module R N
      inst✝ : DecidableEq ι
      p : ι → Submodule R N
      i : ι
      y : N
      hy : Membership.mem (p i) y
      ⊢ Membership.mem (LinearMap.range ((DFinsupp.lsum Nat) fun i => (p i).subtype) …
    -/
    simp only [LinearMap.mem_range, lsum_apply_apply]
    /-
      ι : Type u_1
      R : Type u_2
      N : Type u_5
      inst✝³ : Semiring R
      inst✝² : AddCommMonoid N
      inst✝¹ : Module R N
      inst✝ : DecidableEq ι
      p : ι → Submodule R N
      i : ι
      y : N
      hy : Membership.mem (p i) y
      ⊢ Exists fun y_1 => Eq ((DFinsupp.sumAddHom fun i => (p i).subtype.toAddMonoid …
    -/
    exact ⟨DFinsupp.single i ⟨y, hy⟩, DFinsupp.sumAddHom_single _ _ _⟩
    /-
      🎉 no goals
    -/
    /-
      case a
      ι : Type u_1
      R : Type u_2
      N : Type u_5
      inst✝³ : Semiring R
      inst✝² : AddCommMonoid N
      inst✝¹ : Module R N
      inst✝ : DecidableEq ι
      p : ι → Submodule R N
      ⊢ LE.le (LinearMap.range ((DFinsupp.lsum Nat) fun i => (p i).subtype)) (iSup p)
    -/
  · rintro x ⟨v, rfl⟩
    /-
      case a.intro
      ι : Type u_1
      R : Type u_2
      N : Type u_5
      inst✝³ : Semiring R
      inst✝² : AddCommMonoid N
      inst✝¹ : Module R N
      inst✝ : DecidableEq ι
      p : ι → Submodule R N
      v : DFinsupp fun i => Subtype fun x => Membership.mem (p i) x
      ⊢ Membership.mem (iSup p) (((DFinsupp.lsum Nat) fun i => (p i).subtype) v)
    -/
    exact dfinsupp_sumAddHom_mem _ v _ fun i _ => (le_iSup p i : p i ≤ _) (v i).2
    /-
      🎉 no goals
    -/


/-- The bounded supremum of a family of commutative additive submonoids is equal to the range of
`DFinsupp.sumAddHom` composed with `DFinsupp.filter_add_monoid_hom`; that is, every element in the
bounded `iSup` can be produced from taking a finite number of non-zero elements from the `S i` that
satisfy `p i`, coercing them to `γ`, and summing them. -/
theorem biSup_eq_range_dfinsupp_lsum (p : ι → Prop) [DecidablePred p] (S : ι → Submodule R N) :
    ⨆ (i) (_ : p i), S i =
      LinearMap.range
        (LinearMap.comp
          (DFinsupp.lsum ℕ (M := fun i ↦ ↥(S i)) (fun i => (S i).subtype))
            (DFinsupp.filterLinearMap R _ p)) := by
  /-
    ι : Type u_1
    R : Type u_2
    N : Type u_5
    inst✝⁴ : Semiring R
    inst✝³ : AddCommMonoid N
    inst✝² : Module R N
    inst✝¹ : DecidableEq ι
    p : ι → Prop
    inst✝ : DecidablePred p
    S : ι → Submodule R N
    ⊢ Eq (iSup fun i => iSup fun x => S i) (LinearMap.range (((DFinsupp.lsum Nat)  …
  -/
  apply le_antisymm
    /-
      case a
      ι : Type u_1
      R : Type u_2
      N : Type u_5
      inst✝⁴ : Semiring R
      inst✝³ : AddCommMonoid N
      inst✝² : Module R N
      inst✝¹ : DecidableEq ι
      p : ι → Prop
      inst✝ : DecidablePred p
      S : ι → Submodule R N
      ⊢ LE.le (iSup fun i => iSup fun x => S i) (LinearMap.range (((DFinsupp.lsum Na …
    -/
  · refine iSup₂_le fun i hi y hy => ⟨DFinsupp.single i ⟨y, hy⟩, ?_⟩
    /-
      case a
      ι : Type u_1
      R : Type u_2
      N : Type u_5
      inst✝⁴ : Semiring R
      inst✝³ : AddCommMonoid N
      inst✝² : Module R N
      inst✝¹ : DecidableEq ι
      p : ι → Prop
      inst✝ : DecidablePred p
      S : ι → Submodule R N
      i : ι
      hi : p i
      y : N
      hy : Membership.mem (S i) y
      ⊢ Eq ((((DFinsupp.lsum Nat) fun i => (S i).subtype).comp (DFinsupp.filterLinea …
    -/
    rw [LinearMap.comp_apply, filterLinearMap_apply, filter_single_pos _ _ hi]
    /-
      case a
      ι : Type u_1
      R : Type u_2
      N : Type u_5
      inst✝⁴ : Semiring R
      inst✝³ : AddCommMonoid N
      inst✝² : Module R N
      inst✝¹ : DecidableEq ι
      p : ι → Prop
      inst✝ : DecidablePred p
      S : ι → Submodule R N
      i : ι
      hi : p i
      y : N
      hy : Membership.mem (S i) y
      ⊢ Eq (((DFinsupp.lsum Nat) fun i => (S i).subtype) (DFinsupp.single i ⟨y, hy⟩) …
    -/
    simp only [lsum_apply_apply, sumAddHom_single, LinearMap.toAddMonoidHom_coe, coe_subtype]
    /-
      🎉 no goals
    -/
    /-
      case a
      ι : Type u_1
      R : Type u_2
      N : Type u_5
      inst✝⁴ : Semiring R
      inst✝³ : AddCommMonoid N
      inst✝² : Module R N
      inst✝¹ : DecidableEq ι
      p : ι → Prop
      inst✝ : DecidablePred p
      S : ι → Submodule R N
      ⊢ LE.le (LinearMap.range (((DFinsupp.lsum Nat) fun i => (S i).subtype).comp (D …
    -/
  · rintro x ⟨v, rfl⟩
    /-
      case a.intro
      ι : Type u_1
      R : Type u_2
      N : Type u_5
      inst✝⁴ : Semiring R
      inst✝³ : AddCommMonoid N
      inst✝² : Module R N
      inst✝¹ : DecidableEq ι
      p : ι → Prop
      inst✝ : DecidablePred p
      S : ι → Submodule R N
      v : DFinsupp fun i => Subtype fun x => Membership.mem (S i) x
      ⊢ Membership.mem (iSup fun i => iSup fun x => S i) ((((DFinsupp.lsum Nat) fun  …
    -/
    refine dfinsupp_sumAddHom_mem _ _ _ fun i _ => ?_
    /-
      case a.intro
      ι : Type u_1
      R : Type u_2
      N : Type u_5
      inst✝⁴ : Semiring R
      inst✝³ : AddCommMonoid N
      inst✝² : Module R N
      inst✝¹ : DecidableEq ι
      p : ι → Prop
      inst✝ : DecidablePred p
      S : ι → Submodule R N
      v : DFinsupp fun i => Subtype fun x => Membership.mem (S i) x
      i : ι
      x✝ : Ne (((DFinsupp.filterLinearMap R (fun i => Subtype fun x => Membership.me …
      ⊢ Membership.mem (iSup fun i => iSup fun x => S i) (((fun i => (S i).subtype)  …
    -/
    refine mem_iSup_of_mem i ?_
    /-
      case a.intro
      ι : Type u_1
      R : Type u_2
      N : Type u_5
      inst✝⁴ : Semiring R
      inst✝³ : AddCommMonoid N
      inst✝² : Module R N
      inst✝¹ : DecidableEq ι
      p : ι → Prop
      inst✝ : DecidablePred p
      S : ι → Submodule R N
      v : DFinsupp fun i => Subtype fun x => Membership.mem (S i) x
      i : ι
      x✝ : Ne (((DFinsupp.filterLinearMap R (fun i => Subtype fun x => Membership.me …
      ⊢ Membership.mem (iSup fun x => S i) (((fun i => (S i).subtype) i).toAddMonoid …
    -/
    by_cases hp : p i
      /-
        case pos
        ι : Type u_1
        R : Type u_2
        N : Type u_5
        inst✝⁴ : Semiring R
        inst✝³ : AddCommMonoid N
        inst✝² : Module R N
        inst✝¹ : DecidableEq ι
        p : ι → Prop
        inst✝ : DecidablePred p
        S : ι → Submodule R N
        v : DFinsupp fun i => Subtype fun x => Membership.mem (S i) x
        i : ι
        x✝ : Ne (((DFinsupp.filterLinearMap R (fun i => Subtype fun x => Membership.me …
        hp : p i
        ⊢ Membership.mem (iSup fun x => S i) (((fun i => (S i).subtype) i).toAddMonoid …
      -/
    · simp [hp]
      /-
        🎉 no goals
      -/
      /-
        case neg
        ι : Type u_1
        R : Type u_2
        N : Type u_5
        inst✝⁴ : Semiring R
        inst✝³ : AddCommMonoid N
        inst✝² : Module R N
        inst✝¹ : DecidableEq ι
        p : ι → Prop
        inst✝ : DecidablePred p
        S : ι → Submodule R N
        v : DFinsupp fun i => Subtype fun x => Membership.mem (S i) x
        i : ι
        x✝ : Ne (((DFinsupp.filterLinearMap R (fun i => Subtype fun x => Membership.me …
        hp : Not (p i)
        ⊢ Membership.mem (iSup fun x => S i) (((fun i => (S i).subtype) i).toAddMonoid …
      -/
    · simp [hp]
      /-
        🎉 no goals
      -/


/-- A characterisation of the span of a family of submodules.

See also `Submodule.mem_iSup_iff_exists_finsupp`. -/
theorem mem_iSup_iff_exists_dfinsupp (p : ι → Submodule R N) (x : N) :
    x ∈ iSup p ↔
      ∃ f : Π₀ i, p i, DFinsupp.lsum ℕ (M := fun i ↦ ↥(p i)) (fun i => (p i).subtype) f = x :=
  SetLike.ext_iff.mp (iSup_eq_range_dfinsupp_lsum p) x


/-- A variant of `Submodule.mem_iSup_iff_exists_dfinsupp` with the RHS fully unfolded.

See also `Submodule.mem_iSup_iff_exists_finsupp`. -/
theorem mem_iSup_iff_exists_dfinsupp' (p : ι → Submodule R N) [∀ (i) (x : p i), Decidable (x ≠ 0)]
    (x : N) : x ∈ iSup p ↔ ∃ f : Π₀ i, p i, (f.sum fun _ xi => ↑xi) = x := by
  /-
    ι : Type u_1
    R : Type u_2
    N : Type u_5
    inst✝⁴ : Semiring R
    inst✝³ : AddCommMonoid N
    inst✝² : Module R N
    inst✝¹ : DecidableEq ι
    p : ι → Submodule R N
    inst✝ : (i : ι) → (x : Subtype fun x => Membership.mem (p i) x) → Decidable (N …
    x : N
    ⊢ Iff (Membership.mem (iSup p) x) (Exists fun f => Eq (f.sum fun x xi => ↑xi) x)
  -/
  rw [mem_iSup_iff_exists_dfinsupp]
  simp_rw [DFinsupp.lsum_apply_apply, DFinsupp.sumAddHom_apply,
    LinearMap.toAddMonoidHom_coe, coe_subtype]


theorem mem_biSup_iff_exists_dfinsupp (p : ι → Prop) [DecidablePred p] (S : ι → Submodule R N)
    (x : N) :
    (x ∈ ⨆ (i) (_ : p i), S i) ↔
      ∃ f : Π₀ i, S i,
        DFinsupp.lsum ℕ (M := fun i ↦ ↥(S i)) (fun i => (S i).subtype) (f.filter p) = x :=
  SetLike.ext_iff.mp (biSup_eq_range_dfinsupp_lsum p S) x


lemma mem_iSup_iff_exists_finsupp (p : ι → Submodule R N) (x : N) :
    x ∈ iSup p ↔ ∃ (f : ι →₀ N), (∀ i, f i ∈ p i) ∧ (f.sum fun _i xi ↦ xi) = x := by
  classical
  rw [mem_iSup_iff_exists_dfinsupp']
  refine ⟨fun ⟨f, hf⟩ ↦ ⟨⟨f.support, fun i ↦ (f i : N), by simp⟩, by simp, hf⟩, ?_⟩
  rintro ⟨f, hf, rfl⟩
  refine ⟨DFinsupp.mk f.support fun i ↦ ⟨f i, hf i⟩, Finset.sum_congr ?_ fun i hi ↦ ?_⟩
  · ext; simp [mk_eq_zero]
  · simp [Finsupp.mem_support_iff.mp hi]


theorem mem_iSup_finset_iff_exists_sum {s : Finset ι} (p : ι → Submodule R N) (a : N) :
    (a ∈ ⨆ i ∈ s, p i) ↔ ∃ μ : ∀ i, p i, (∑ i ∈ s, (μ i : N)) = a := by
  classical
    rw [Submodule.mem_iSup_iff_exists_dfinsupp']
    constructor <;> rintro ⟨μ, hμ⟩
    · use fun i => ⟨μ i, (iSup_const_le : _ ≤ p i) (coe_mem <| μ i)⟩
      rw [← hμ]
      symm
      apply Finset.sum_subset
      · intro x
        contrapose
        intro hx
        rw [mem_support_iff, not_ne_iff]
        ext
        rw [coe_zero, ← mem_bot R]
        suffices ⊥ = ⨆ (_ : x ∈ s), p x from this.symm ▸ coe_mem (μ x)
        exact (iSup_neg hx).symm
      · intro x _ hx
        rw [mem_support_iff, not_ne_iff] at hx
        rw [hx]
        rfl
    · refine ⟨DFinsupp.mk s ?_, ?_⟩
      · rintro ⟨i, hi⟩
        refine ⟨μ i, ?_⟩
        rw [iSup_pos]
        · exact coe_mem _
        · exact hi
      simp only [DFinsupp.sum]
      rw [Finset.sum_subset support_mk_subset, ← hμ]
      · exact Finset.sum_congr rfl fun x hx => by rw [mk_of_mem hx]
      · intro x _ hx
        rw [mem_support_iff, not_ne_iff] at hx
        rw [hx]
        rfl


/-- Independence of a family of submodules can be expressed as a quantifier over `DFinsupp`s.

This is an intermediate result used to prove
`iSupIndep_of_dfinsupp_lsum_injective` and
`iSupIndep.dfinsupp_lsum_injective`. -/
theorem iSupIndep_iff_forall_dfinsupp (p : ι → Submodule R N) :
    iSupIndep p ↔
      ∀ (i) (x : p i) (v : Π₀ i : ι, ↥(p i)),
        lsum ℕ (M := fun i ↦ ↥(p i)) (fun i => (p i).subtype) (erase i v) = x → x = 0 := by
  simp_rw [iSupIndep_def, Submodule.disjoint_def,
    Submodule.mem_biSup_iff_exists_dfinsupp, exists_imp, filter_ne_eq_erase]
  /-
    ι : Type u_1
    R : Type u_2
    N : Type u_5
    inst✝³ : DecidableEq ι
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid N
    inst✝ : Module R N
    p : ι → Submodule R N
    ⊢ Iff (∀ (i : ι) (x : N), Membership.mem (p i) x → ∀ (x_1 : DFinsupp fun i =>  …
  -/
  refine forall_congr' fun i => Subtype.forall'.trans ?_
  /-
    ι : Type u_1
    R : Type u_2
    N : Type u_5
    inst✝³ : DecidableEq ι
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid N
    inst✝ : Module R N
    p : ι → Submodule R N
    i : ι
    ⊢ Iff (∀ (x : Subtype fun a => Membership.mem (p i) a) (x_1 : DFinsupp fun i = …
  -/
  simp_rw [Submodule.coe_eq_zero]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-11-24")]
alias independent_iff_forall_dfinsupp := iSupIndep_iff_forall_dfinsupp

/- If `DFinsupp.lsum` applied with `Submodule.subtype` is injective then the submodules are
iSupIndep. -/

theorem iSupIndep_of_dfinsupp_lsum_injective (p : ι → Submodule R N)
    (h : Function.Injective (lsum ℕ (M := fun i ↦ ↥(p i)) fun i => (p i).subtype)) :
    iSupIndep p := by
  /-
    ι : Type u_1
    R : Type u_2
    N : Type u_5
    inst✝³ : DecidableEq ι
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid N
    inst✝ : Module R N
    p : ι → Submodule R N
    h : Function.Injective ⇑((DFinsupp.lsum Nat) fun i => (p i).subtype)
    ⊢ iSupIndep p
  -/
  rw [iSupIndep_iff_forall_dfinsupp]
  /-
    ι : Type u_1
    R : Type u_2
    N : Type u_5
    inst✝³ : DecidableEq ι
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid N
    inst✝ : Module R N
    p : ι → Submodule R N
    h : Function.Injective ⇑((DFinsupp.lsum Nat) fun i => (p i).subtype)
    ⊢ ∀ (i : ι) (x : Subtype fun x => Membership.mem (p i) x) (v : DFinsupp fun i  …
  -/
  intro i x v hv
  replace hv : lsum ℕ (M := fun i ↦ ↥(p i)) (fun i => (p i).subtype) (erase i v) =
      lsum ℕ (M := fun i ↦ ↥(p i)) (fun i => (p i).subtype) (single i x) := by
    simpa only [lsum_single] using hv
  /-
    ι : Type u_1
    R : Type u_2
    N : Type u_5
    inst✝³ : DecidableEq ι
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid N
    inst✝ : Module R N
    p : ι → Submodule R N
    h : Function.Injective ⇑((DFinsupp.lsum Nat) fun i => (p i).subtype)
    i : ι
    x : Subtype fun x => Membership.mem (p i) x
    v : DFinsupp fun i => Subtype fun x => Membership.mem (p i) x
    hv : Eq (((DFinsupp.lsum Nat) fun i => (p i).subtype) (DFinsupp.erase i v)) (( …
    ⊢ Eq x 0
  -/
  have := DFunLike.ext_iff.mp (h hv) i
  /-
    ι : Type u_1
    R : Type u_2
    N : Type u_5
    inst✝³ : DecidableEq ι
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid N
    inst✝ : Module R N
    p : ι → Submodule R N
    h : Function.Injective ⇑((DFinsupp.lsum Nat) fun i => (p i).subtype)
    i : ι
    x : Subtype fun x => Membership.mem (p i) x
    v : DFinsupp fun i => Subtype fun x => Membership.mem (p i) x
    hv : Eq (((DFinsupp.lsum Nat) fun i => (p i).subtype) (DFinsupp.erase i v)) (( …
    this : Eq ((DFinsupp.erase i v) i) ((DFinsupp.single i x) i)
    ⊢ Eq x 0
  -/
  simpa [eq_comm] using this
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-11-24")]
alias independent_of_dfinsupp_lsum_injective := iSupIndep_of_dfinsupp_lsum_injective

/- If `DFinsupp.sumAddHom` applied with `AddSubmonoid.subtype` is injective then the additive
submonoids are independent. -/

theorem iSupIndep_of_dfinsupp_sumAddHom_injective (p : ι → AddSubmonoid N)
    (h : Function.Injective (sumAddHom fun i => (p i).subtype)) : iSupIndep p := by
  /-
    ι : Type u_1
    N : Type u_5
    inst✝¹ : DecidableEq ι
    inst✝ : AddCommMonoid N
    p : ι → AddSubmonoid N
    h : Function.Injective ⇑(DFinsupp.sumAddHom fun i => (p i).subtype)
    ⊢ iSupIndep p
  -/
  rw [← iSupIndep_map_orderIso_iff (AddSubmonoid.toNatSubmodule : AddSubmonoid N ≃o _)]
  /-
    ι : Type u_1
    N : Type u_5
    inst✝¹ : DecidableEq ι
    inst✝ : AddCommMonoid N
    p : ι → AddSubmonoid N
    h : Function.Injective ⇑(DFinsupp.sumAddHom fun i => (p i).subtype)
    ⊢ iSupIndep (Function.comp (⇑AddSubmonoid.toNatSubmodule) p)
  -/
  exact iSupIndep_of_dfinsupp_lsum_injective _ h
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-11-24")]
alias independent_of_dfinsupp_sumAddHom_injective := iSupIndep_of_dfinsupp_sumAddHom_injective


/-- Combining `DFinsupp.lsum` with `LinearMap.toSpanSingleton` is the same as
`Finsupp.linearCombination` -/
theorem lsum_comp_mapRange_toSpanSingleton [∀ m : R, Decidable (m ≠ 0)] (p : ι → Submodule R N)
    {v : ι → N} (hv : ∀ i : ι, v i ∈ p i) :
    (lsum ℕ (M := fun i ↦ ↥(p i)) fun i => (p i).subtype : _ →ₗ[R] _).comp
        ((mapRange.linearMap fun i => LinearMap.toSpanSingleton R (↥(p i)) ⟨v i, hv i⟩ :
              _ →ₗ[R] _).comp
          (finsuppLequivDFinsupp R : (ι →₀ R) ≃ₗ[R] _).toLinearMap) =
      Finsupp.linearCombination R v := by
  /-
    ι : Type u_1
    R : Type u_2
    N : Type u_5
    inst✝⁴ : DecidableEq ι
    inst✝³ : Semiring R
    inst✝² : AddCommMonoid N
    inst✝¹ : Module R N
    inst✝ : (m : R) → Decidable (Ne m 0)
    p : ι → Submodule R N
    v : ι → N
    hv : ∀ (i : ι), Membership.mem (p i) (v i)
    ⊢ Eq (((DFinsupp.lsum Nat) fun i => (p i).subtype).comp ((DFinsupp.mapRange.li …
  -/
  ext
  /-
    case h.h
    ι : Type u_1
    R : Type u_2
    N : Type u_5
    inst✝⁴ : DecidableEq ι
    inst✝³ : Semiring R
    inst✝² : AddCommMonoid N
    inst✝¹ : Module R N
    inst✝ : (m : R) → Decidable (Ne m 0)
    p : ι → Submodule R N
    v : ι → N
    hv : ∀ (i : ι), Membership.mem (p i) (v i)
    a✝ : ι
    ⊢ Eq (((((DFinsupp.lsum Nat) fun i => (p i).subtype).comp ((DFinsupp.mapRange. …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- If `DFinsupp.sumAddHom` applied with `AddSubmonoid.subtype` is injective then the additive
subgroups are independent. -/
theorem iSupIndep_of_dfinsupp_sumAddHom_injective' (p : ι → AddSubgroup N)
    (h : Function.Injective (sumAddHom fun i => (p i).subtype)) : iSupIndep p := by
  /-
    ι : Type u_1
    N : Type u_5
    inst✝¹ : DecidableEq ι
    inst✝ : AddCommGroup N
    p : ι → AddSubgroup N
    h : Function.Injective ⇑(DFinsupp.sumAddHom fun i => (p i).subtype)
    ⊢ iSupIndep p
  -/
  rw [← iSupIndep_map_orderIso_iff (AddSubgroup.toIntSubmodule : AddSubgroup N ≃o _)]
  /-
    ι : Type u_1
    N : Type u_5
    inst✝¹ : DecidableEq ι
    inst✝ : AddCommGroup N
    p : ι → AddSubgroup N
    h : Function.Injective ⇑(DFinsupp.sumAddHom fun i => (p i).subtype)
    ⊢ iSupIndep (Function.comp (⇑AddSubgroup.toIntSubmodule) p)
  -/
  exact iSupIndep_of_dfinsupp_lsum_injective _ h
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-11-24")]
alias independent_of_dfinsupp_sumAddHom_injective' := iSupIndep_of_dfinsupp_sumAddHom_injective'


/-- The canonical map out of a direct sum of a family of submodules is injective when the submodules
are `iSupIndep`.

Note that this is not generally true for `[Semiring R]`, for instance when `A` is the
`ℕ`-submodules of the positive and negative integers.

See `Counterexamples/DirectSumIsInternal.lean` for a proof of this fact. -/
theorem iSupIndep.dfinsupp_lsum_injective {p : ι → Submodule R N} (h : iSupIndep p) :
    Function.Injective (lsum ℕ (M := fun i ↦ ↥(p i)) fun i => (p i).subtype) := by
  -- simplify everything down to binders over equalities in `N`
  /-
    ι : Type u_1
    R : Type u_2
    N : Type u_5
    inst✝³ : DecidableEq ι
    inst✝² : Ring R
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    p : ι → Submodule R N
    h : iSupIndep p
    ⊢ Function.Injective ⇑((DFinsupp.lsum Nat) fun i => (p i).subtype)
  -/
  rw [iSupIndep_iff_forall_dfinsupp] at h
  suffices LinearMap.ker (lsum ℕ (M := fun i ↦ ↥(p i)) fun i => (p i).subtype) = ⊥ by
    -- Lean can't find this without our help
    letI thisI : AddCommGroup (Π₀ i, p i) := inferInstance
    rw [LinearMap.ker_eq_bot] at this
    exact this
  /-
    ι : Type u_1
    R : Type u_2
    N : Type u_5
    inst✝³ : DecidableEq ι
    inst✝² : Ring R
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    p : ι → Submodule R N
    h : ∀ (i : ι) (x : Subtype fun x => Membership.mem (p i) x) (v : DFinsupp fun  …
    ⊢ Eq (LinearMap.ker ((DFinsupp.lsum Nat) fun i => (p i).subtype)) Bot.bot
  -/
  rw [LinearMap.ker_eq_bot']
  /-
    ι : Type u_1
    R : Type u_2
    N : Type u_5
    inst✝³ : DecidableEq ι
    inst✝² : Ring R
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    p : ι → Submodule R N
    h : ∀ (i : ι) (x : Subtype fun x => Membership.mem (p i) x) (v : DFinsupp fun  …
    ⊢ ∀ (m : DFinsupp fun i => Subtype fun x => Membership.mem (p i) x), Eq (((DFi …
  -/
  intro m hm
  /-
    ι : Type u_1
    R : Type u_2
    N : Type u_5
    inst✝³ : DecidableEq ι
    inst✝² : Ring R
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    p : ι → Submodule R N
    h : ∀ (i : ι) (x : Subtype fun x => Membership.mem (p i) x) (v : DFinsupp fun  …
    m : DFinsupp fun i => Subtype fun x => Membership.mem (p i) x
    hm : Eq (((DFinsupp.lsum Nat) fun i => (p i).subtype) m) 0
    ⊢ Eq m 0
  -/
  ext i : 1
  -- split `m` into the piece at `i` and the pieces elsewhere, to match `h`
  /-
    case h
    ι : Type u_1
    R : Type u_2
    N : Type u_5
    inst✝³ : DecidableEq ι
    inst✝² : Ring R
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    p : ι → Submodule R N
    h : ∀ (i : ι) (x : Subtype fun x => Membership.mem (p i) x) (v : DFinsupp fun  …
    m : DFinsupp fun i => Subtype fun x => Membership.mem (p i) x
    hm : Eq (((DFinsupp.lsum Nat) fun i => (p i).subtype) m) 0
    i : ι
    ⊢ Eq (m i) (0 i)
  -/
  rw [DFinsupp.zero_apply, ← neg_eq_zero]
  /-
    case h
    ι : Type u_1
    R : Type u_2
    N : Type u_5
    inst✝³ : DecidableEq ι
    inst✝² : Ring R
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    p : ι → Submodule R N
    h : ∀ (i : ι) (x : Subtype fun x => Membership.mem (p i) x) (v : DFinsupp fun  …
    m : DFinsupp fun i => Subtype fun x => Membership.mem (p i) x
    hm : Eq (((DFinsupp.lsum Nat) fun i => (p i).subtype) m) 0
    i : ι
    ⊢ Eq (Neg.neg (m i)) 0
  -/
  refine h i (-m i) m ?_
  rwa [← erase_add_single i m, LinearMap.map_add, lsum_single, Submodule.subtype_apply,
    add_eq_zero_iff_eq_neg, ← Submodule.coe_neg] at hm


@[deprecated (since := "2024-11-24")]
alias Independent.dfinsupp_lsum_injective := iSupIndep.dfinsupp_lsum_injective


/-- The canonical map out of a direct sum of a family of additive subgroups is injective when the
additive subgroups are `iSupIndep`. -/
theorem iSupIndep.dfinsupp_sumAddHom_injective {p : ι → AddSubgroup N} (h : iSupIndep p) :
    Function.Injective (sumAddHom fun i => (p i).subtype) := by
  /-
    ι : Type u_1
    N : Type u_5
    inst✝¹ : DecidableEq ι
    inst✝ : AddCommGroup N
    p : ι → AddSubgroup N
    h : iSupIndep p
    ⊢ Function.Injective ⇑(DFinsupp.sumAddHom fun i => (p i).subtype)
  -/
  rw [← iSupIndep_map_orderIso_iff (AddSubgroup.toIntSubmodule : AddSubgroup N ≃o _)] at h
  /-
    ι : Type u_1
    N : Type u_5
    inst✝¹ : DecidableEq ι
    inst✝ : AddCommGroup N
    p : ι → AddSubgroup N
    h : iSupIndep (Function.comp (⇑AddSubgroup.toIntSubmodule) p)
    ⊢ Function.Injective ⇑(DFinsupp.sumAddHom fun i => (p i).subtype)
  -/
  exact h.dfinsupp_lsum_injective
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-11-24")]
alias Independent.dfinsupp_sumAddHom_injective := iSupIndep.dfinsupp_sumAddHom_injective


/-- A family of submodules over an additive group are independent if and only iff `DFinsupp.lsum`
applied with `Submodule.subtype` is injective.

Note that this is not generally true for `[Semiring R]`; see
`iSupIndep.dfinsupp_lsum_injective` for details. -/
theorem iSupIndep_iff_dfinsupp_lsum_injective (p : ι → Submodule R N) :
    iSupIndep p ↔ Function.Injective (lsum ℕ (M := fun i ↦ ↥(p i)) fun i => (p i).subtype) :=
  ⟨iSupIndep.dfinsupp_lsum_injective, iSupIndep_of_dfinsupp_lsum_injective p⟩


@[deprecated (since := "2024-11-24")]
alias independent_iff_dfinsupp_lsum_injective := iSupIndep_iff_dfinsupp_lsum_injective


/-- A family of additive subgroups over an additive group are independent if and only if
`DFinsupp.sumAddHom` applied with `AddSubgroup.subtype` is injective. -/
theorem iSupIndep_iff_dfinsupp_sumAddHom_injective (p : ι → AddSubgroup N) :
    iSupIndep p ↔ Function.Injective (sumAddHom fun i => (p i).subtype) :=
  ⟨iSupIndep.dfinsupp_sumAddHom_injective, iSupIndep_of_dfinsupp_sumAddHom_injective' p⟩


@[deprecated (since := "2024-11-24")]
alias independent_iff_dfinsupp_sumAddHom_injective := iSupIndep_iff_dfinsupp_sumAddHom_injective


/-- If a family of submodules is independent, then a choice of nonzero vector from each submodule
forms a linearly independent family.

See also `iSupIndep.linearIndependent'`. -/
theorem iSupIndep.linearIndependent [NoZeroSMulDivisors R N] {ι} (p : ι → Submodule R N)
    (hp : iSupIndep p) {v : ι → N} (hv : ∀ i, v i ∈ p i) (hv' : ∀ i, v i ≠ 0) :
    LinearIndependent R v := by
  /-
    R : Type u_2
    N : Type u_5
    inst✝³ : Ring R
    inst✝² : AddCommGroup N
    inst✝¹ : Module R N
    inst✝ : NoZeroSMulDivisors R N
    ι : Type u_6
    p : ι → Submodule R N
    hp : iSupIndep p
    v : ι → N
    hv : ∀ (i : ι), Membership.mem (p i) (v i)
    hv' : ∀ (i : ι), Ne (v i) 0
    ⊢ LinearIndependent R v
  -/
  let _ := Classical.decEq ι
  /-
    R : Type u_2
    N : Type u_5
    inst✝³ : Ring R
    inst✝² : AddCommGroup N
    inst✝¹ : Module R N
    inst✝ : NoZeroSMulDivisors R N
    ι : Type u_6
    p : ι → Submodule R N
    hp : iSupIndep p
    v : ι → N
    hv : ∀ (i : ι), Membership.mem (p i) (v i)
    hv' : ∀ (i : ι), Ne (v i) 0
    x✝ : DecidableEq ι := Classical.decEq ι
    ⊢ LinearIndependent R v
  -/
  let _ := Classical.decEq R
  /-
    R : Type u_2
    N : Type u_5
    inst✝³ : Ring R
    inst✝² : AddCommGroup N
    inst✝¹ : Module R N
    inst✝ : NoZeroSMulDivisors R N
    ι : Type u_6
    p : ι → Submodule R N
    hp : iSupIndep p
    v : ι → N
    hv : ∀ (i : ι), Membership.mem (p i) (v i)
    hv' : ∀ (i : ι), Ne (v i) 0
    x✝¹ : DecidableEq ι := Classical.decEq ι
    x✝ : DecidableEq R := Classical.decEq R
    ⊢ LinearIndependent R v
  -/
  rw [linearIndependent_iff]
  /-
    R : Type u_2
    N : Type u_5
    inst✝³ : Ring R
    inst✝² : AddCommGroup N
    inst✝¹ : Module R N
    inst✝ : NoZeroSMulDivisors R N
    ι : Type u_6
    p : ι → Submodule R N
    hp : iSupIndep p
    v : ι → N
    hv : ∀ (i : ι), Membership.mem (p i) (v i)
    hv' : ∀ (i : ι), Ne (v i) 0
    x✝¹ : DecidableEq ι := Classical.decEq ι
    x✝ : DecidableEq R := Classical.decEq R
    ⊢ ∀ (l : Finsupp ι R), Eq ((Finsupp.linearCombination R v) l) 0 → Eq l 0
  -/
  intro l hl
  let a :=
    DFinsupp.mapRange.linearMap (fun i => LinearMap.toSpanSingleton R (p i) ⟨v i, hv i⟩)
      l.toDFinsupp
  have ha : a = 0 := by
    apply hp.dfinsupp_lsum_injective
    rwa [← lsum_comp_mapRange_toSpanSingleton _ hv] at hl
  /-
    R : Type u_2
    N : Type u_5
    inst✝³ : Ring R
    inst✝² : AddCommGroup N
    inst✝¹ : Module R N
    inst✝ : NoZeroSMulDivisors R N
    ι : Type u_6
    p : ι → Submodule R N
    hp : iSupIndep p
    v : ι → N
    hv : ∀ (i : ι), Membership.mem (p i) (v i)
    hv' : ∀ (i : ι), Ne (v i) 0
    x✝¹ : DecidableEq ι := Classical.decEq ι
    x✝ : DecidableEq R := Classical.decEq R
    l : Finsupp ι R
    hl : Eq ((Finsupp.linearCombination R v) l) 0
    a : DFinsupp fun i => Subtype fun x => Membership.mem (p i) x := (DFinsupp.map …
    ha : Eq a 0
    ⊢ Eq l 0
  -/
  ext i
  /-
    case h
    R : Type u_2
    N : Type u_5
    inst✝³ : Ring R
    inst✝² : AddCommGroup N
    inst✝¹ : Module R N
    inst✝ : NoZeroSMulDivisors R N
    ι : Type u_6
    p : ι → Submodule R N
    hp : iSupIndep p
    v : ι → N
    hv : ∀ (i : ι), Membership.mem (p i) (v i)
    hv' : ∀ (i : ι), Ne (v i) 0
    x✝¹ : DecidableEq ι := Classical.decEq ι
    x✝ : DecidableEq R := Classical.decEq R
    l : Finsupp ι R
    hl : Eq ((Finsupp.linearCombination R v) l) 0
    a : DFinsupp fun i => Subtype fun x => Membership.mem (p i) x := (DFinsupp.map …
    ha : Eq a 0
    i : ι
    ⊢ Eq (l i) (0 i)
  -/
  apply smul_left_injective R (hv' i)
  /-
    case h.a
    R : Type u_2
    N : Type u_5
    inst✝³ : Ring R
    inst✝² : AddCommGroup N
    inst✝¹ : Module R N
    inst✝ : NoZeroSMulDivisors R N
    ι : Type u_6
    p : ι → Submodule R N
    hp : iSupIndep p
    v : ι → N
    hv : ∀ (i : ι), Membership.mem (p i) (v i)
    hv' : ∀ (i : ι), Ne (v i) 0
    x✝¹ : DecidableEq ι := Classical.decEq ι
    x✝ : DecidableEq R := Classical.decEq R
    l : Finsupp ι R
    hl : Eq ((Finsupp.linearCombination R v) l) 0
    a : DFinsupp fun i => Subtype fun x => Membership.mem (p i) x := (DFinsupp.map …
    ha : Eq a 0
    i : ι
    ⊢ Eq ((fun c => HSMul.hSMul c (v i)) (l i)) ((fun c => HSMul.hSMul c (v i)) (0 …
  -/
  have : l i • v i = a i := rfl
  /-
    case h.a
    R : Type u_2
    N : Type u_5
    inst✝³ : Ring R
    inst✝² : AddCommGroup N
    inst✝¹ : Module R N
    inst✝ : NoZeroSMulDivisors R N
    ι : Type u_6
    p : ι → Submodule R N
    hp : iSupIndep p
    v : ι → N
    hv : ∀ (i : ι), Membership.mem (p i) (v i)
    hv' : ∀ (i : ι), Ne (v i) 0
    x✝¹ : DecidableEq ι := Classical.decEq ι
    x✝ : DecidableEq R := Classical.decEq R
    l : Finsupp ι R
    hl : Eq ((Finsupp.linearCombination R v) l) 0
    a : DFinsupp fun i => Subtype fun x => Membership.mem (p i) x := (DFinsupp.map …
    ha : Eq a 0
    i : ι
    this : Eq (HSMul.hSMul (l i) (v i)) ↑(a i)
    ⊢ Eq ((fun c => HSMul.hSMul c (v i)) (l i)) ((fun c => HSMul.hSMul c (v i)) (0 …
  -/
  simp only [coe_zero, Pi.zero_apply, ZeroMemClass.coe_zero, smul_eq_zero, ha] at this
  /-
    case h.a
    R : Type u_2
    N : Type u_5
    inst✝³ : Ring R
    inst✝² : AddCommGroup N
    inst✝¹ : Module R N
    inst✝ : NoZeroSMulDivisors R N
    ι : Type u_6
    p : ι → Submodule R N
    hp : iSupIndep p
    v : ι → N
    hv : ∀ (i : ι), Membership.mem (p i) (v i)
    hv' : ∀ (i : ι), Ne (v i) 0
    x✝¹ : DecidableEq ι := Classical.decEq ι
    x✝ : DecidableEq R := Classical.decEq R
    l : Finsupp ι R
    hl : Eq ((Finsupp.linearCombination R v) l) 0
    a : DFinsupp fun i => Subtype fun x => Membership.mem (p i) x := (DFinsupp.map …
    ha : Eq a 0
    i : ι
    this : Or (Eq (l i) 0) (Eq (v i) 0)
    ⊢ Eq ((fun c => HSMul.hSMul c (v i)) (l i)) ((fun c => HSMul.hSMul c (v i)) (0 …
  -/
  simpa
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-11-24")]
alias Independent.linearIndependent := iSupIndep.linearIndependent


theorem iSupIndep_iff_linearIndependent_of_ne_zero [NoZeroSMulDivisors R N] {ι} {v : ι → N}
    (h_ne_zero : ∀ i, v i ≠ 0) : (iSupIndep fun i => R ∙ v i) ↔ LinearIndependent R v :=
  let _ := Classical.decEq ι
  ⟨fun hv => hv.linearIndependent _ (fun i => Submodule.mem_span_singleton_self <| v i) h_ne_zero,
    fun hv => hv.iSupIndep_span_singleton⟩


@[deprecated (since := "2024-11-24")]
alias independent_iff_linearIndependent_of_ne_zero := iSupIndep_iff_linearIndependent_of_ne_zero


theorem coe_dfinsupp_sum (t : Π₀ i, γ i) (g : ∀ i, γ i → M →ₛₗ[σ₁₂] M₂) :
    ⇑(t.sum g) = t.sum fun i d => g i d := rfl


@[simp]
theorem dfinsupp_sum_apply (t : Π₀ i, γ i) (g : ∀ i, γ i → M →ₛₗ[σ₁₂] M₂) (b : M) :
    (t.sum g) b = t.sum fun i d => g i d b :=
  sum_apply _ _ _


@[simp]
theorem map_dfinsupp_sumAddHom (f : M →ₛₗ[σ₁₂] M₂) {t : Π₀ i, γ i} {g : ∀ i, γ i →+ M} :
    f (sumAddHom g t) = sumAddHom (fun i => f.toAddMonoidHom.comp (g i)) t :=
  f.toAddMonoidHom.map_dfinsupp_sumAddHom _ _


@[simp]
theorem map_dfinsupp_sumAddHom [∀ i, AddZeroClass (γ i)] (f : M ≃ₛₗ[τ₁₂] M₂) (t : Π₀ i, γ i)
    (g : ∀ i, γ i →+ M) :
    f (sumAddHom g t) = sumAddHom (fun i => f.toAddEquiv.toAddMonoidHom.comp (g i)) t :=
  f.toAddEquiv.map_dfinsupp_sumAddHom _ _


