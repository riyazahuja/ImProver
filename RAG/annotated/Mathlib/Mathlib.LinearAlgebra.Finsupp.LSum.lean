theorem smul_sum [Zero β] [AddCommMonoid M] [DistribSMul R M] {v : α →₀ β} {c : R} {h : α → β → M} :
    c • v.sum h = v.sum fun a b => c • h a b :=
  Finset.smul_sum


@[simp]
theorem sum_smul_index_linearMap' [Semiring R] [AddCommMonoid M] [Module R M] [AddCommMonoid M₂]
    [Module R M₂] {v : α →₀ M} {c : R} {h : α → M →ₗ[R] M₂} :
    ((c • v).sum fun a => h a) = c • v.sum fun a => h a := by
  /-
    α : Type u_1
    R : Type u_3
    M : Type u_4
    M₂ : Type u_5
    inst✝⁴ : Semiring R
    inst✝³ : AddCommMonoid M
    inst✝² : Module R M
    inst✝¹ : AddCommMonoid M₂
    inst✝ : Module R M₂
    v : Finsupp α M
    c : R
    h : α → LinearMap (RingHom.id R) M M₂
    ⊢ Eq ((HSMul.hSMul c v).sum fun a => ⇑(h a)) (HSMul.hSMul c (v.sum fun a => ⇑( …
  -/
  rw [Finsupp.sum_smul_index', Finsupp.smul_sum]
    /-
      α : Type u_1
      R : Type u_3
      M : Type u_4
      M₂ : Type u_5
      inst✝⁴ : Semiring R
      inst✝³ : AddCommMonoid M
      inst✝² : Module R M
      inst✝¹ : AddCommMonoid M₂
      inst✝ : Module R M₂
      v : Finsupp α M
      c : R
      h : α → LinearMap (RingHom.id R) M M₂
      ⊢ Eq (v.sum fun i c_1 => (h i) (HSMul.hSMul c c_1)) (v.sum fun a b => HSMul.hS …
    -/
  · simp only [map_smul]
    /-
      🎉 no goals
    -/
    /-
      α : Type u_1
      R : Type u_3
      M : Type u_4
      M₂ : Type u_5
      inst✝⁴ : Semiring R
      inst✝³ : AddCommMonoid M
      inst✝² : Module R M
      inst✝¹ : AddCommMonoid M₂
      inst✝ : Module R M₂
      v : Finsupp α M
      c : R
      h : α → LinearMap (RingHom.id R) M M₂
      ⊢ ∀ (i : α), Eq ((h i) 0) 0
    -/
  · intro i
    /-
      α : Type u_1
      R : Type u_3
      M : Type u_4
      M₂ : Type u_5
      inst✝⁴ : Semiring R
      inst✝³ : AddCommMonoid M
      inst✝² : Module R M
      inst✝¹ : AddCommMonoid M₂
      inst✝ : Module R M₂
      v : Finsupp α M
      c : R
      h : α → LinearMap (RingHom.id R) M M₂
      i : α
      ⊢ Eq ((h i) 0) 0
    -/
    exact (h i).map_zero
    /-
      🎉 no goals
    -/


instance _root_.LinearMap.CompatibleSMul.finsupp_dom [SMulZeroClass R M] [DistribSMul R N]
    [LinearMap.CompatibleSMul M N R S] : LinearMap.CompatibleSMul (ι →₀ M) N R S where
  map_smul f r m := by
    /-
      α : Type u_1
      M✝ : Type u_2
      N✝ : Type u_3
      P : Type u_4
      R✝ : Type u_5
      S✝ : Type u_6
      inst✝¹⁵ : Semiring R✝
      inst✝¹⁴ : Semiring S✝
      inst✝¹³ : AddCommMonoid M✝
      inst✝¹² : Module R✝ M✝
      inst✝¹¹ : AddCommMonoid N✝
      inst✝¹⁰ : Module R✝ N✝
      inst✝⁹ : AddCommMonoid P
      inst✝⁸ : Module R✝ P
      R : Type u_7
      S : Type u_8
      M : Type u_9
      N : Type u_10
      ι : Type u_11
      inst✝⁷ : Semiring S
      inst✝⁶ : AddCommMonoid M
      inst✝⁵ : AddCommMonoid N
      inst✝⁴ : Module S M
      inst✝³ : Module S N
      inst✝² : SMulZeroClass R M
      inst✝¹ : DistribSMul R N
      inst✝ : LinearMap.CompatibleSMul M N R S
      f : LinearMap (RingHom.id S) (Finsupp ι M) N
      r : R
      m : Finsupp ι M
      ⊢ Eq (f (HSMul.hSMul r m)) (HSMul.hSMul r (f m))
    -/
    conv_rhs => rw [← sum_single m, map_finsupp_sum, smul_sum]
    /-
      α : Type u_1
      M✝ : Type u_2
      N✝ : Type u_3
      P : Type u_4
      R✝ : Type u_5
      S✝ : Type u_6
      inst✝¹⁵ : Semiring R✝
      inst✝¹⁴ : Semiring S✝
      inst✝¹³ : AddCommMonoid M✝
      inst✝¹² : Module R✝ M✝
      inst✝¹¹ : AddCommMonoid N✝
      inst✝¹⁰ : Module R✝ N✝
      inst✝⁹ : AddCommMonoid P
      inst✝⁸ : Module R✝ P
      R : Type u_7
      S : Type u_8
      M : Type u_9
      N : Type u_10
      ι : Type u_11
      inst✝⁷ : Semiring S
      inst✝⁶ : AddCommMonoid M
      inst✝⁵ : AddCommMonoid N
      inst✝⁴ : Module S M
      inst✝³ : Module S N
      inst✝² : SMulZeroClass R M
      inst✝¹ : DistribSMul R N
      inst✝ : LinearMap.CompatibleSMul M N R S
      f : LinearMap (RingHom.id S) (Finsupp ι M) N
      r : R
      m : Finsupp ι M
      ⊢ Eq (f (HSMul.hSMul r m)) (m.sum fun a b => HSMul.hSMul r (f (Finsupp.single  …
    -/
    erw [← sum_single (r • m), sum_mapRange_index single_zero, map_finsupp_sum]
    /-
      α : Type u_1
      M✝ : Type u_2
      N✝ : Type u_3
      P : Type u_4
      R✝ : Type u_5
      S✝ : Type u_6
      inst✝¹⁵ : Semiring R✝
      inst✝¹⁴ : Semiring S✝
      inst✝¹³ : AddCommMonoid M✝
      inst✝¹² : Module R✝ M✝
      inst✝¹¹ : AddCommMonoid N✝
      inst✝¹⁰ : Module R✝ N✝
      inst✝⁹ : AddCommMonoid P
      inst✝⁸ : Module R✝ P
      R : Type u_7
      S : Type u_8
      M : Type u_9
      N : Type u_10
      ι : Type u_11
      inst✝⁷ : Semiring S
      inst✝⁶ : AddCommMonoid M
      inst✝⁵ : AddCommMonoid N
      inst✝⁴ : Module S M
      inst✝³ : Module S N
      inst✝² : SMulZeroClass R M
      inst✝¹ : DistribSMul R N
      inst✝ : LinearMap.CompatibleSMul M N R S
      f : LinearMap (RingHom.id S) (Finsupp ι M) N
      r : R
      m : Finsupp ι M
      ⊢ Eq (m.sum fun a b => f (Finsupp.single a (HSMul.hSMul r b))) (m.sum fun a b  …
    -/
    congr; ext i m; exact (f.comp <| lsingle i).map_smul_of_tower r m
                    /-
                      🎉 no goals
                    -/


instance _root_.LinearMap.CompatibleSMul.finsupp_cod [SMul R M] [SMulZeroClass R N]
    [LinearMap.CompatibleSMul M N R S] : LinearMap.CompatibleSMul M (ι →₀ N) R S where
                       /-
                         α : Type u_1
                         M✝ : Type u_2
                         N✝ : Type u_3
                         P : Type u_4
                         R✝ : Type u_5
                         S✝ : Type u_6
                         inst✝¹⁵ : Semiring R✝
                         inst✝¹⁴ : Semiring S✝
                         inst✝¹³ : AddCommMonoid M✝
                         inst✝¹² : Module R✝ M✝
                         inst✝¹¹ : AddCommMonoid N✝
                         inst✝¹⁰ : Module R✝ N✝
                         inst✝⁹ : AddCommMonoid P
                         inst✝⁸ : Module R✝ P
                         R : Type u_7
                         S : Type u_8
                         M : Type u_9
                         N : Type u_10
                         ι : Type u_11
                         inst✝⁷ : Semiring S
                         inst✝⁶ : AddCommMonoid M
                         inst✝⁵ : AddCommMonoid N
                         inst✝⁴ : Module S M
                         inst✝³ : Module S N
                         inst✝² : SMul R M
                         inst✝¹ : SMulZeroClass R N
                         inst✝ : LinearMap.CompatibleSMul M N R S
                         f : LinearMap (RingHom.id S) M (Finsupp ι N)
                         r : R
                         m : M
                         ⊢ Eq (f (HSMul.hSMul r m)) (HSMul.hSMul r (f m))
                       -/
  map_smul f r m := by ext i; apply ((lapply i).comp f).map_smul_of_tower
                              /-
                                🎉 no goals
                              -/


/-- Lift a family of linear maps `M →ₗ[R] N` indexed by `x : α` to a linear map from `α →₀ M` to
`N` using `Finsupp.sum`. This is an upgraded version of `Finsupp.liftAddHom`.

See note [bundled maps over different rings] for why separate `R` and `S` semirings are used.
-/
def lsum : (α → M →ₗ[R] N) ≃ₗ[S] (α →₀ M) →ₗ[R] N where
  toFun F :=
    { toFun := fun d => d.sum fun i => F i
      map_add' := (liftAddHom (α := α) (M := M) (N := N) fun x => (F x).toAddMonoidHom).map_add
                                 /-
                                   α : Type u_1
                                   M : Type u_2
                                   N : Type u_3
                                   P : Type u_4
                                   R : Type u_5
                                   S : Type u_6
                                   inst✝⁹ : Semiring R
                                   inst✝⁸ : Semiring S
                                   inst✝⁷ : AddCommMonoid M
                                   inst✝⁶ : Module R M
                                   inst✝⁵ : AddCommMonoid N
                                   inst✝⁴ : Module R N
                                   inst✝³ : AddCommMonoid P
                                   inst✝² : Module R P
                                   inst✝¹ : Module S N
                                   inst✝ : SMulCommClass R S N
                                   F : α → LinearMap (RingHom.id R) M N
                                   c : R
                                   f : Finsupp α M
                                   ⊢ Eq ({ toFun := fun d => d.sum fun i => ⇑(F i), map_add' := ⋯ }.toFun (HSMul. …
                                 -/
      map_smul' := fun c f => by simp [sum_smul_index', smul_sum] }
                                 /-
                                   🎉 no goals
                                 -/
  invFun F x := F.comp (lsingle x)
  left_inv F := by
    /-
      α : Type u_1
      M : Type u_2
      N : Type u_3
      P : Type u_4
      R : Type u_5
      S : Type u_6
      inst✝⁹ : Semiring R
      inst✝⁸ : Semiring S
      inst✝⁷ : AddCommMonoid M
      inst✝⁶ : Module R M
      inst✝⁵ : AddCommMonoid N
      inst✝⁴ : Module R N
      inst✝³ : AddCommMonoid P
      inst✝² : Module R P
      inst✝¹ : Module S N
      inst✝ : SMulCommClass R S N
      F : α → LinearMap (RingHom.id R) M N
      ⊢ Eq ((fun F x => F.comp (Finsupp.lsingle x)) ({ toFun := fun F => { toFun :=  …
    -/
    ext x y
    /-
      case h.h
      α : Type u_1
      M : Type u_2
      N : Type u_3
      P : Type u_4
      R : Type u_5
      S : Type u_6
      inst✝⁹ : Semiring R
      inst✝⁸ : Semiring S
      inst✝⁷ : AddCommMonoid M
      inst✝⁶ : Module R M
      inst✝⁵ : AddCommMonoid N
      inst✝⁴ : Module R N
      inst✝³ : AddCommMonoid P
      inst✝² : Module R P
      inst✝¹ : Module S N
      inst✝ : SMulCommClass R S N
      F : α → LinearMap (RingHom.id R) M N
      x : α
      y : M
      ⊢ Eq (((fun F x => F.comp (Finsupp.lsingle x)) ({ toFun := fun F => { toFun := …
    -/
    simp
    /-
      🎉 no goals
    -/
  right_inv F := by
    /-
      α : Type u_1
      M : Type u_2
      N : Type u_3
      P : Type u_4
      R : Type u_5
      S : Type u_6
      inst✝⁹ : Semiring R
      inst✝⁸ : Semiring S
      inst✝⁷ : AddCommMonoid M
      inst✝⁶ : Module R M
      inst✝⁵ : AddCommMonoid N
      inst✝⁴ : Module R N
      inst✝³ : AddCommMonoid P
      inst✝² : Module R P
      inst✝¹ : Module S N
      inst✝ : SMulCommClass R S N
      F G : α → LinearMap (RingHom.id R) M N
      ⊢ Eq ((fun F => { toFun := fun d => d.sum fun i => ⇑(F i), map_add' := ⋯, map_ …
    -/
    /-
      α : Type u_1
      M : Type u_2
      N : Type u_3
      P : Type u_4
      R : Type u_5
      S : Type u_6
      inst✝⁹ : Semiring R
      inst✝⁸ : Semiring S
      inst✝⁷ : AddCommMonoid M
      inst✝⁶ : Module R M
      inst✝⁵ : AddCommMonoid N
      inst✝⁴ : Module R N
      inst✝³ : AddCommMonoid P
      inst✝² : Module R P
      inst✝¹ : Module S N
      inst✝ : SMulCommClass R S N
      F : LinearMap (RingHom.id R) (Finsupp α M) N
      ⊢ Eq ({ toFun := fun F => { toFun := fun d => d.sum fun i => ⇑(F i), map_add'  …
    -/
    /-
      case h.h
      α : Type u_1
      M : Type u_2
      N : Type u_3
      P : Type u_4
      R : Type u_5
      S : Type u_6
      inst✝⁹ : Semiring R
      inst✝⁸ : Semiring S
      inst✝⁷ : AddCommMonoid M
      inst✝⁶ : Module R M
      inst✝⁵ : AddCommMonoid N
      inst✝⁴ : Module R N
      inst✝³ : AddCommMonoid P
      inst✝² : Module R P
      inst✝¹ : Module S N
      inst✝ : SMulCommClass R S N
      F G : α → LinearMap (RingHom.id R) M N
      x : α
      y : M
      ⊢ Eq ((((fun F => { toFun := fun d => d.sum fun i => ⇑(F i), map_add' := ⋯, ma …
    -/
    ext x y
    /-
      🎉 no goals
    -/
    /-
      case h.h
      α : Type u_1
      M : Type u_2
      N : Type u_3
      P : Type u_4
      R : Type u_5
      S : Type u_6
      inst✝⁹ : Semiring R
      inst✝⁸ : Semiring S
      inst✝⁷ : AddCommMonoid M
      inst✝⁶ : Module R M
      inst✝⁵ : AddCommMonoid N
      inst✝⁴ : Module R N
      inst✝³ : AddCommMonoid P
      inst✝² : Module R P
      inst✝¹ : Module S N
      inst✝ : SMulCommClass R S N
      F : LinearMap (RingHom.id R) (Finsupp α M) N
      x : α
      y : M
      ⊢ Eq ((({ toFun := fun F => { toFun := fun d => d.sum fun i => ⇑(F i), map_add …
    -/
    /-
      α : Type u_1
      M : Type u_2
      N : Type u_3
      P : Type u_4
      R : Type u_5
      S : Type u_6
      inst✝⁹ : Semiring R
      inst✝⁸ : Semiring S
      inst✝⁷ : AddCommMonoid M
      inst✝⁶ : Module R M
      inst✝⁵ : AddCommMonoid N
      inst✝⁴ : Module R N
      inst✝³ : AddCommMonoid P
      inst✝² : Module R P
      inst✝¹ : Module S N
      inst✝ : SMulCommClass R S N
      F : S
      G : α → LinearMap (RingHom.id R) M N
      ⊢ Eq ({ toFun := fun F => { toFun := fun d => d.sum fun i => ⇑(F i), map_add'  …
    -/
    simp
    /-
      case h.h
      α : Type u_1
      M : Type u_2
      N : Type u_3
      P : Type u_4
      R : Type u_5
      S : Type u_6
      inst✝⁹ : Semiring R
      inst✝⁸ : Semiring S
      inst✝⁷ : AddCommMonoid M
      inst✝⁶ : Module R M
      inst✝⁵ : AddCommMonoid N
      inst✝⁴ : Module R N
      inst✝³ : AddCommMonoid P
      inst✝² : Module R P
      inst✝¹ : Module S N
      inst✝ : SMulCommClass R S N
      F : S
      G : α → LinearMap (RingHom.id R) M N
      x : α
      y : M
      ⊢ Eq ((({ toFun := fun F => { toFun := fun d => d.sum fun i => ⇑(F i), map_add …
    -/
    /-
      🎉 no goals
    -/
    /-
      🎉 no goals
    -/
  map_add' F G := by
    ext x y
    simp
  map_smul' F G := by
    ext x y
    simp


@[simp]
theorem coe_lsum (f : α → M →ₗ[R] N) : (lsum S f : (α →₀ M) → N) = fun d => d.sum fun i => f i :=
  rfl


theorem lsum_apply (f : α → M →ₗ[R] N) (l : α →₀ M) : Finsupp.lsum S f l = l.sum fun b => f b :=
  rfl


theorem lsum_single (f : α → M →ₗ[R] N) (i : α) (m : M) :
    Finsupp.lsum S f (Finsupp.single i m) = f i m :=
  Finsupp.sum_single_index (f i).map_zero


@[simp] theorem lsum_comp_lsingle (f : α → M →ₗ[R] N) (i : α) :
                                              /-
                                                α : Type u_1
                                                M : Type u_2
                                                N : Type u_3
                                                R : Type u_5
                                                S : Type u_6
                                                inst✝⁷ : Semiring R
                                                inst✝⁶ : Semiring S
                                                inst✝⁵ : AddCommMonoid M
                                                inst✝⁴ : Module R M
                                                inst✝³ : AddCommMonoid N
                                                inst✝² : Module R N
                                                inst✝¹ : Module S N
                                                inst✝ : SMulCommClass R S N
                                                f : α → LinearMap (RingHom.id R) M N
                                                i : α
                                                ⊢ Eq (((Finsupp.lsum S) f).comp (Finsupp.lsingle i)) (f i)
                                              -/
    Finsupp.lsum S f ∘ₗ lsingle i = f i := by ext; simp
                                                   /-
                                                     🎉 no goals
                                                   -/


theorem lsum_symm_apply (f : (α →₀ M) →ₗ[R] N) (x : α) : (lsum S).symm f x = f.comp (lsingle x) :=
  rfl


/-- A slight rearrangement from `lsum` gives us
the bijection underlying the free-forgetful adjunction for R-modules.
-/
noncomputable def lift : (X → M) ≃+ ((X →₀ R) →ₗ[R] M) :=
  (AddEquiv.arrowCongr (Equiv.refl X) (ringLmapEquivSelf R ℕ M).toAddEquiv.symm).trans
    (lsum _ : _ ≃ₗ[ℕ] _).toAddEquiv


@[simp]
theorem lift_symm_apply (f) (x) : ((lift M R X).symm f) x = f (single x 1) :=
  rfl


@[simp]
theorem lift_apply (f) (g) : ((lift M R X) f) g = g.sum fun x r => r • f x :=
  rfl


/-- Given compatible `S` and `R`-module structures on `M` and a type `X`, the set of functions
`X → M` is `S`-linearly equivalent to the `R`-linear maps from the free `R`-module
on `X` to `M`. -/
noncomputable def llift : (X → M) ≃ₗ[S] (X →₀ R) →ₗ[R] M :=
  { lift M R X with
    map_smul' := by
      /-
        α : Type u_1
        M : Type u_2
        N : Type u_3
        P : Type u_4
        R : Type u_5
        S : Type u_6
        inst✝⁹ : Semiring R
        inst✝⁸ : Semiring S
        inst✝⁷ : AddCommMonoid M
        inst✝⁶ : Module R M
        inst✝⁵ : AddCommMonoid N
        inst✝⁴ : Module R N
        inst✝³ : AddCommMonoid P
        inst✝² : Module R P
        X : Type u_7
        inst✝¹ : Module S M
        inst✝ : SMulCommClass R S M
        ⊢ ∀ (m : S) (x : X → M), Eq ({ toFun := __src✝.toFun, map_add' := ⋯ }.toFun (H …
      -/
      intros
      /-
        α : Type u_1
        M : Type u_2
        N : Type u_3
        P : Type u_4
        R : Type u_5
        S : Type u_6
        inst✝⁹ : Semiring R
        inst✝⁸ : Semiring S
        inst✝⁷ : AddCommMonoid M
        inst✝⁶ : Module R M
        inst✝⁵ : AddCommMonoid N
        inst✝⁴ : Module R N
        inst✝³ : AddCommMonoid P
        inst✝² : Module R P
        X : Type u_7
        inst✝¹ : Module S M
        inst✝ : SMulCommClass R S M
        m✝ : S
        x✝ : X → M
        ⊢ Eq ({ toFun := __src✝.toFun, map_add' := ⋯ }.toFun (HSMul.hSMul m✝ x✝)) (HSM …
      -/
      dsimp
      /-
        α : Type u_1
        M : Type u_2
        N : Type u_3
        P : Type u_4
        R : Type u_5
        S : Type u_6
        inst✝⁹ : Semiring R
        inst✝⁸ : Semiring S
        inst✝⁷ : AddCommMonoid M
        inst✝⁶ : Module R M
        inst✝⁵ : AddCommMonoid N
        inst✝⁴ : Module R N
        inst✝³ : AddCommMonoid P
        inst✝² : Module R P
        X : Type u_7
        inst✝¹ : Module S M
        inst✝ : SMulCommClass R S M
        m✝ : S
        x✝ : X → M
        ⊢ Eq ((Finsupp.lift M R X) (HSMul.hSMul m✝ x✝)) (HSMul.hSMul m✝ ((Finsupp.lift …
      -/
      ext
      simp only [coe_comp, Function.comp_apply, lsingle_apply, lift_apply, Pi.smul_apply,
        sum_single_index, zero_smul, one_smul, LinearMap.smul_apply] }


@[simp]
theorem llift_apply (f : X → M) (x : X →₀ R) : llift M R S X f x = lift M R X f x :=
  rfl


@[simp]
theorem llift_symm_apply (f : (X →₀ R) →ₗ[R] M) (x : X) :
    (llift M R S X).symm f x = f (single x 1) :=
  rfl


/-- An equivalence of domains induces a linear equivalence of finitely supported functions.

This is `Finsupp.domCongr` as a `LinearEquiv`.
See also `LinearMap.funCongrLeft` for the case of arbitrary functions. -/
protected def domLCongr {α₁ α₂ : Type*} (e : α₁ ≃ α₂) : (α₁ →₀ M) ≃ₗ[R] α₂ →₀ M :=
  (Finsupp.domCongr e : (α₁ →₀ M) ≃+ (α₂ →₀ M)).toLinearEquiv <| by
    /-
      α : Type u_1
      M : Type u_2
      N : Type u_3
      P : Type u_4
      R : Type u_5
      S : Type u_6
      inst✝⁷ : Semiring R
      inst✝⁶ : Semiring S
      inst✝⁵ : AddCommMonoid M
      inst✝⁴ : Module R M
      inst✝³ : AddCommMonoid N
      inst✝² : Module R N
      inst✝¹ : AddCommMonoid P
      inst✝ : Module R P
      α₁ : Type u_7
      α₂ : Type u_8
      e : Equiv α₁ α₂
      ⊢ ∀ (c : R) (x : Finsupp α₁ M), Eq ((Finsupp.domCongr e) (HSMul.hSMul c x)) (H …
    -/
    simpa only [equivMapDomain_eq_mapDomain, domCongr_apply] using (lmapDomain M R e).map_smul
    /-
      🎉 no goals
    -/


@[simp]
theorem domLCongr_apply {α₁ : Type*} {α₂ : Type*} (e : α₁ ≃ α₂) (v : α₁ →₀ M) :
    (Finsupp.domLCongr e : _ ≃ₗ[R] _) v = Finsupp.domCongr e v :=
  rfl


@[simp]
theorem domLCongr_refl : Finsupp.domLCongr (Equiv.refl α) = LinearEquiv.refl R (α →₀ M) :=
  LinearEquiv.ext fun _ => equivMapDomain_refl _


theorem domLCongr_trans {α₁ α₂ α₃ : Type*} (f : α₁ ≃ α₂) (f₂ : α₂ ≃ α₃) :
    (Finsupp.domLCongr f).trans (Finsupp.domLCongr f₂) =
      (Finsupp.domLCongr (f.trans f₂) : (_ →₀ M) ≃ₗ[R] _) :=
  LinearEquiv.ext fun _ => (equivMapDomain_trans _ _ _).symm


@[simp]
theorem domLCongr_symm {α₁ α₂ : Type*} (f : α₁ ≃ α₂) :
    ((Finsupp.domLCongr f).symm : (_ →₀ M) ≃ₗ[R] _) = Finsupp.domLCongr f.symm :=
  LinearEquiv.ext fun _ => rfl


theorem domLCongr_single {α₁ : Type*} {α₂ : Type*} (e : α₁ ≃ α₂) (i : α₁) (m : M) :
    (Finsupp.domLCongr e : _ ≃ₗ[R] _) (Finsupp.single i m) = Finsupp.single (e i) m := by
  /-
    M : Type u_2
    R : Type u_5
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    α₁ : Type u_7
    α₂ : Type u_8
    e : Equiv α₁ α₂
    i : α₁
    m : M
    ⊢ Eq ((Finsupp.domLCongr e) (Finsupp.single i m)) (Finsupp.single (e i) m)
  -/
  simp
  /-
    🎉 no goals
  -/


/-- An equivalence of domain and a linear equivalence of codomain induce a linear equivalence of the
corresponding finitely supported functions. -/
def lcongr {ι κ : Sort _} (e₁ : ι ≃ κ) (e₂ : M ≃ₗ[R] N) : (ι →₀ M) ≃ₗ[R] κ →₀ N :=
  (Finsupp.domLCongr e₁).trans (mapRange.linearEquiv e₂)


@[simp]
theorem lcongr_single {ι κ : Sort _} (e₁ : ι ≃ κ) (e₂ : M ≃ₗ[R] N) (i : ι) (m : M) :
                                                                           /-
                                                                             M : Type u_2
                                                                             N : Type u_3
                                                                             R : Type u_5
                                                                             inst✝⁴ : Semiring R
                                                                             inst✝³ : AddCommMonoid M
                                                                             inst✝² : Module R M
                                                                             inst✝¹ : AddCommMonoid N
                                                                             inst✝ : Module R N
                                                                             ι : Type u_7
                                                                             κ : Type u_8
                                                                             e₁ : Equiv ι κ
                                                                             e₂ : LinearEquiv (RingHom.id R) M N
                                                                             i : ι
                                                                             m : M
                                                                             ⊢ Eq ((Finsupp.lcongr e₁ e₂) (Finsupp.single i m)) (Finsupp.single (e₁ i) (e₂  …
                                                                           -/
    lcongr e₁ e₂ (Finsupp.single i m) = Finsupp.single (e₁ i) (e₂ m) := by simp [lcongr]
                                                                           /-
                                                                             🎉 no goals
                                                                           -/


@[simp]
theorem lcongr_apply_apply {ι κ : Sort _} (e₁ : ι ≃ κ) (e₂ : M ≃ₗ[R] N) (f : ι →₀ M) (k : κ) :
    lcongr e₁ e₂ f k = e₂ (f (e₁.symm k)) :=
  rfl


theorem lcongr_symm_single {ι κ : Sort _} (e₁ : ι ≃ κ) (e₂ : M ≃ₗ[R] N) (k : κ) (n : N) :
    (lcongr e₁ e₂).symm (Finsupp.single k n) = Finsupp.single (e₁.symm k) (e₂.symm n) := by
  /-
    M : Type u_2
    N : Type u_3
    R : Type u_5
    inst✝⁴ : Semiring R
    inst✝³ : AddCommMonoid M
    inst✝² : Module R M
    inst✝¹ : AddCommMonoid N
    inst✝ : Module R N
    ι : Type u_7
    κ : Type u_8
    e₁ : Equiv ι κ
    e₂ : LinearEquiv (RingHom.id R) M N
    k : κ
    n : N
    ⊢ Eq ((Finsupp.lcongr e₁ e₂).symm (Finsupp.single k n)) (Finsupp.single (e₁.sy …
  -/
  apply_fun (lcongr e₁ e₂ : (ι →₀ M) → (κ →₀ N)) using (lcongr e₁ e₂).injective
  /-
    M : Type u_2
    N : Type u_3
    R : Type u_5
    inst✝⁴ : Semiring R
    inst✝³ : AddCommMonoid M
    inst✝² : Module R M
    inst✝¹ : AddCommMonoid N
    inst✝ : Module R N
    ι : Type u_7
    κ : Type u_8
    e₁ : Equiv ι κ
    e₂ : LinearEquiv (RingHom.id R) M N
    k : κ
    n : N
    ⊢ Eq ((Finsupp.lcongr e₁ e₂) ((Finsupp.lcongr e₁ e₂).symm (Finsupp.single k n) …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem lcongr_symm {ι κ : Sort _} (e₁ : ι ≃ κ) (e₂ : M ≃ₗ[R] N) :
    (lcongr e₁ e₂).symm = lcongr e₁.symm e₂.symm := by
  /-
    M : Type u_2
    N : Type u_3
    R : Type u_5
    inst✝⁴ : Semiring R
    inst✝³ : AddCommMonoid M
    inst✝² : Module R M
    inst✝¹ : AddCommMonoid N
    inst✝ : Module R N
    ι : Type u_7
    κ : Type u_8
    e₁ : Equiv ι κ
    e₂ : LinearEquiv (RingHom.id R) M N
    ⊢ Eq (Finsupp.lcongr e₁ e₂).symm (Finsupp.lcongr e₁.symm e₂.symm)
  -/
  ext
  /-
    case h.h
    M : Type u_2
    N : Type u_3
    R : Type u_5
    inst✝⁴ : Semiring R
    inst✝³ : AddCommMonoid M
    inst✝² : Module R M
    inst✝¹ : AddCommMonoid N
    inst✝ : Module R N
    ι : Type u_7
    κ : Type u_8
    e₁ : Equiv ι κ
    e₂ : LinearEquiv (RingHom.id R) M N
    x✝ : Finsupp κ N
    a✝ : ι
    ⊢ Eq (((Finsupp.lcongr e₁ e₂).symm x✝) a✝) (((Finsupp.lcongr e₁.symm e₂.symm)  …
  -/
  rfl
  /-
    🎉 no goals
  -/


protected theorem Submodule.finsupp_sum_mem {ι β : Type*} [Zero β] (S : Submodule R M) (f : ι →₀ β)
    (g : ι → β → M) (h : ∀ c, f c ≠ 0 → g c (f c) ∈ S) : f.sum g ∈ S :=
  AddSubmonoidClass.finsupp_sum_mem S f g h


/-- A surjective linear map to finitely supported functions has a splitting. -/
def splittingOfFinsuppSurjective (f : M →ₗ[R] α →₀ R) (s : Surjective f) : (α →₀ R) →ₗ[R] M :=
  Finsupp.lift _ _ _ fun x : α => (s (Finsupp.single x 1)).choose


theorem splittingOfFinsuppSurjective_splits (f : M →ₗ[R] α →₀ R) (s : Surjective f) :
    f.comp (splittingOfFinsuppSurjective f s) = LinearMap.id := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    α : Type u_4
    f : LinearMap (RingHom.id R) M (Finsupp α R)
    s : Function.Surjective ⇑f
    ⊢ Eq (f.comp (f.splittingOfFinsuppSurjective s)) LinearMap.id
  -/
  ext x
  /-
    case h.h.h
    R : Type u_1
    M : Type u_2
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    α : Type u_4
    f : LinearMap (RingHom.id R) M (Finsupp α R)
    s : Function.Surjective ⇑f
    x a✝ : α
    ⊢ Eq ((((f.comp (f.splittingOfFinsuppSurjective s)).comp (Finsupp.lsingle x))  …
  -/
  dsimp [splittingOfFinsuppSurjective]
  /-
    case h.h.h
    R : Type u_1
    M : Type u_2
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    α : Type u_4
    f : LinearMap (RingHom.id R) M (Finsupp α R)
    s : Function.Surjective ⇑f
    x a✝ : α
    ⊢ Eq ((f ((Finsupp.single x 1).sum fun x r => HSMul.hSMul r ⋯.choose)) a✝) ((F …
  -/
  congr
  /-
    case h.h.h.e_a
    R : Type u_1
    M : Type u_2
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    α : Type u_4
    f : LinearMap (RingHom.id R) M (Finsupp α R)
    s : Function.Surjective ⇑f
    x a✝ : α
    ⊢ Eq (f ((Finsupp.single x 1).sum fun x r => HSMul.hSMul r ⋯.choose)) (Finsupp …
  -/
  rw [sum_single_index, one_smul]
    /-
      case h.h.h.e_a
      R : Type u_1
      M : Type u_2
      inst✝² : Semiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      α : Type u_4
      f : LinearMap (RingHom.id R) M (Finsupp α R)
      s : Function.Surjective ⇑f
      x a✝ : α
      ⊢ Eq (f ⋯.choose) (Finsupp.single x 1)
    -/
  · exact (s (Finsupp.single x 1)).choose_spec
    /-
      🎉 no goals
    -/
    /-
      case h.h.h.e_a
      R : Type u_1
      M : Type u_2
      inst✝² : Semiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      α : Type u_4
      f : LinearMap (RingHom.id R) M (Finsupp α R)
      s : Function.Surjective ⇑f
      x a✝ : α
      ⊢ Eq (HSMul.hSMul 0 ⋯.choose) 0
    -/
  · rw [zero_smul]
    /-
      🎉 no goals
    -/


theorem leftInverse_splittingOfFinsuppSurjective (f : M →ₗ[R] α →₀ R) (s : Surjective f) :
    LeftInverse f (splittingOfFinsuppSurjective f s) := fun g =>
  LinearMap.congr_fun (splittingOfFinsuppSurjective_splits f s) g


theorem splittingOfFinsuppSurjective_injective (f : M →ₗ[R] α →₀ R) (s : Surjective f) :
    Injective (splittingOfFinsuppSurjective f s) :=
  (leftInverse_splittingOfFinsuppSurjective f s).injective


theorem coe_finsupp_sum (t : ι →₀ γ) (g : ι → γ → M →ₛₗ[σ₁₂] M₂) :
    ⇑(t.sum g) = t.sum fun i d => g i d := rfl


@[simp]
theorem finsupp_sum_apply (t : ι →₀ γ) (g : ι → γ → M →ₛₗ[σ₁₂] M₂) (b : M) :
    (t.sum g) b = t.sum fun i d => g i d b :=
  sum_apply _ _ _


/-- If `M` and `N` are submodules of an `R`-algebra `S`, `m : ι → M` is a family of elements, then
there is an `R`-linear map from `ι →₀ N` to `S` which maps `{ n_i }` to the sum of `m_i * n_i`.
This is used in the definition of linearly disjointness. -/
def mulLeftMap {M : Submodule R S} (N : Submodule R S) {ι : Type*} (m : ι → M) :
    (ι →₀ N) →ₗ[R] S := Finsupp.lsum R fun i ↦ (m i).1 • N.subtype


theorem mulLeftMap_apply {M N : Submodule R S} {ι : Type*} (m : ι → M) (n : ι →₀ N) :
    mulLeftMap N m n = Finsupp.sum n fun (i : ι) (n : N) ↦ (m i).1 * n.1 := rfl


@[simp]
theorem mulLeftMap_apply_single {M N : Submodule R S} {ι : Type*} (m : ι → M) (i : ι) (n : N) :
    mulLeftMap N m (Finsupp.single i n) = (m i).1 * n.1 := by
  /-
    R : Type u_1
    inst✝⁴ : Semiring R
    S : Type u_4
    inst✝³ : Semiring S
    inst✝² : Module R S
    inst✝¹ : SMulCommClass R R S
    inst✝ : SMulCommClass R S S
    M N : Submodule R S
    ι : Type u_5
    m : ι → Subtype fun x => Membership.mem M x
    i : ι
    n : Subtype fun x => Membership.mem N x
    ⊢ Eq ((Submodule.mulLeftMap N m) (Finsupp.single i n)) (HMul.hMul ↑(m i) ↑n)
  -/
  simp [mulLeftMap]
  /-
    🎉 no goals
  -/


/-- If `M` and `N` are submodules of an `R`-algebra `S`, `n : ι → N` is a family of elements, then
there is an `R`-linear map from `ι →₀ M` to `S` which maps `{ m_i }` to the sum of `m_i * n_i`.
This is used in the definition of linearly disjointness. -/
def mulRightMap (M : Submodule R S) {N : Submodule R S} {ι : Type*} (n : ι → N) :
    (ι →₀ M) →ₗ[R] S := Finsupp.lsum R fun i ↦ MulOpposite.op (n i).1 • M.subtype


theorem mulRightMap_apply {M N : Submodule R S} {ι : Type*} (n : ι → N) (m : ι →₀ M) :
    mulRightMap M n m = Finsupp.sum m fun (i : ι) (m : M) ↦ m.1 * (n i).1 := rfl


@[simp]
theorem mulRightMap_apply_single {M N : Submodule R S} {ι : Type*} (n : ι → N) (i : ι) (m : M) :
    mulRightMap M n (Finsupp.single i m) = m.1 * (n i).1 := by
  /-
    R : Type u_1
    inst✝⁴ : Semiring R
    S : Type u_4
    inst✝³ : Semiring S
    inst✝² : Module R S
    inst✝¹ : SMulCommClass R R S
    inst✝ : IsScalarTower R S S
    M N : Submodule R S
    ι : Type u_5
    n : ι → Subtype fun x => Membership.mem N x
    i : ι
    m : Subtype fun x => Membership.mem M x
    ⊢ Eq ((M.mulRightMap n) (Finsupp.single i m)) (HMul.hMul ↑m ↑(n i))
  -/
  simp [mulRightMap]
  /-
    🎉 no goals
  -/


theorem mulLeftMap_eq_mulRightMap_of_commute [SMulCommClass R S S]
    {M : Submodule R S} (N : Submodule R S) {ι : Type*} (m : ι → M)
    (hc : ∀ (i : ι) (n : N), Commute (m i).1 n.1) : mulLeftMap N m = mulRightMap N m := by
  /-
    R : Type u_1
    inst✝⁵ : Semiring R
    S : Type u_4
    inst✝⁴ : Semiring S
    inst✝³ : Module R S
    inst✝² : SMulCommClass R R S
    inst✝¹ : IsScalarTower R S S
    inst✝ : SMulCommClass R S S
    M N : Submodule R S
    ι : Type u_5
    m : ι → Subtype fun x => Membership.mem M x
    hc : ∀ (i : ι) (n : Subtype fun x => Membership.mem N x), Commute ↑(m i) ↑n
    ⊢ Eq (Submodule.mulLeftMap N m) (N.mulRightMap m)
  -/
  ext i n; simp [(hc i n).eq]
           /-
             🎉 no goals
           -/


theorem mulLeftMap_eq_mulRightMap {S : Type*} [CommSemiring S] [Module R S] [SMulCommClass R R S]
    [SMulCommClass R S S] [IsScalarTower R S S] {M : Submodule R S} (N : Submodule R S)
    {ι : Type*} (m : ι → M) : mulLeftMap N m = mulRightMap N m :=
  mulLeftMap_eq_mulRightMap_of_commute N m fun _ _ ↦ mul_comm _ _


