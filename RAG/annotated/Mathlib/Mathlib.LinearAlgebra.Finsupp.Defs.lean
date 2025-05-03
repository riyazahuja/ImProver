/-- Given `Finite α`, `linearEquivFunOnFinite R` is the natural `R`-linear equivalence between
`α →₀ β` and `α → β`. -/
@[simps apply]
noncomputable def linearEquivFunOnFinite : (α →₀ M) ≃ₗ[R] α → M :=
  { equivFunOnFinite with
    toFun := (⇑)
    map_add' := fun _ _ => rfl
    map_smul' := fun _ _ => rfl }


@[simp]
theorem linearEquivFunOnFinite_single [DecidableEq α] (x : α) (m : M) :
    (linearEquivFunOnFinite R M α) (single x m) = Pi.single x m :=
  equivFunOnFinite_single x m


@[simp]
theorem linearEquivFunOnFinite_symm_single [DecidableEq α] (x : α) (m : M) :
    (linearEquivFunOnFinite R M α).symm (Pi.single x m) = single x m :=
  equivFunOnFinite_symm_single x m


@[simp]
theorem linearEquivFunOnFinite_symm_coe (f : α →₀ M) : (linearEquivFunOnFinite R M α).symm f = f :=
  (linearEquivFunOnFinite R M α).symm_apply_apply f


/-- Interpret `Finsupp.single a` as a linear map. -/
def lsingle (a : α) : M →ₗ[R] α →₀ M :=
  { Finsupp.singleAddHom a with map_smul' := fun _ _ => (smul_single _ _ _).symm }


/-- Two `R`-linear maps from `Finsupp X M` which agree on each `single x y` agree everywhere. -/
theorem lhom_ext ⦃φ ψ : (α →₀ M) →ₗ[R] N⦄ (h : ∀ a b, φ (single a b) = ψ (single a b)) : φ = ψ :=
  LinearMap.toAddMonoidHom_injective <| addHom_ext h


/-- Two `R`-linear maps from `Finsupp X M` which agree on each `single x y` agree everywhere.

We formulate this fact using equality of linear maps `φ.comp (lsingle a)` and `ψ.comp (lsingle a)`
so that the `ext` tactic can apply a type-specific extensionality lemma to prove equality of these
maps. E.g., if `M = R`, then it suffices to verify `φ (single a 1) = ψ (single a 1)`. -/
-- Porting note: The priority should be higher than `LinearMap.ext`.
@[ext high]
theorem lhom_ext' ⦃φ ψ : (α →₀ M) →ₗ[R] N⦄ (h : ∀ a, φ.comp (lsingle a) = ψ.comp (lsingle a)) :
    φ = ψ :=
  lhom_ext fun a => LinearMap.congr_fun (h a)


/-- Interpret `fun f : α →₀ M ↦ f a` as a linear map. -/
def lapply (a : α) : (α →₀ M) →ₗ[R] M :=
  { Finsupp.applyAddHom a with map_smul' := fun _ _ => rfl }


/-- Interpret `Finsupp.subtypeDomain s` as a linear map. -/
def lsubtypeDomain : (α →₀ M) →ₗ[R] s →₀ M where
  toFun := subtypeDomain fun x => x ∈ s
  map_add' _ _ := subtypeDomain_add
  map_smul' _ _ := ext fun _ => rfl


theorem lsubtypeDomain_apply (f : α →₀ M) :
    (lsubtypeDomain s : (α →₀ M) →ₗ[R] s →₀ M) f = subtypeDomain (fun x => x ∈ s) f :=
  rfl


@[simp]
theorem lsingle_apply (a : α) (b : M) : (lsingle a : M →ₗ[R] α →₀ M) b = single a b :=
  rfl


@[simp]
theorem lapply_apply (a : α) (f : α →₀ M) : (lapply a : (α →₀ M) →ₗ[R] M) f = f a :=
  rfl


@[simp]
                                                                                           /-
                                                                                             α : Type u_1
                                                                                             M : Type u_2
                                                                                             R : Type u_5
                                                                                             inst✝² : Semiring R
                                                                                             inst✝¹ : AddCommMonoid M
                                                                                             inst✝ : Module R M
                                                                                             a : α
                                                                                             ⊢ Eq ((Finsupp.lapply a).comp (Finsupp.lsingle a)) LinearMap.id
                                                                                           -/
theorem lapply_comp_lsingle_same (a : α) : lapply a ∘ₗ lsingle a = (.id : M →ₗ[R] M) := by ext; simp
                                                                                                /-
                                                                                                  🎉 no goals
                                                                                                -/


@[simp]
theorem lapply_comp_lsingle_of_ne (a a' : α) (h : a ≠ a') :
                                                   /-
                                                     α : Type u_1
                                                     M : Type u_2
                                                     R : Type u_5
                                                     inst✝² : Semiring R
                                                     inst✝¹ : AddCommMonoid M
                                                     inst✝ : Module R M
                                                     a a' : α
                                                     h : Ne a a'
                                                     ⊢ Eq ((Finsupp.lapply a).comp (Finsupp.lsingle a')) 0
                                                   -/
    lapply a ∘ₗ lsingle a' = (0 : M →ₗ[R] M) := by ext; simp [h.symm]
                                                        /-
                                                          🎉 no goals
                                                        -/


/-- Interpret `Finsupp.mapDomain` as a linear map. -/
def lmapDomain (f : α → α') : (α →₀ M) →ₗ[R] α' →₀ M where
  toFun := mapDomain f
  map_add' _ _ := mapDomain_add
  map_smul' := mapDomain_smul


@[simp]
theorem lmapDomain_apply (f : α → α') (l : α →₀ M) :
    (lmapDomain M R f : (α →₀ M) →ₗ[R] α' →₀ M) l = mapDomain f l :=
  rfl


@[simp]
theorem lmapDomain_id : (lmapDomain M R _root_.id : (α →₀ M) →ₗ[R] α →₀ M) = LinearMap.id :=
  LinearMap.ext fun _ => mapDomain_id


theorem lmapDomain_comp (f : α → α') (g : α' → α'') :
    lmapDomain M R (g ∘ f) = (lmapDomain M R g).comp (lmapDomain M R f) :=
  LinearMap.ext fun _ => mapDomain_comp


/-- Given `f : α → β` and a proof `hf` that `f` is injective, `lcomapDomain f hf` is the linear map
sending `l : β →₀ M` to the finitely supported function from `α` to `M` given by composing
`l` with `f`.

This is the linear version of `Finsupp.comapDomain`. -/
def lcomapDomain (f : α → β) (hf : Function.Injective f) : (β →₀ M) →ₗ[R] α →₀ M where
  toFun l := Finsupp.comapDomain f l hf.injOn
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
                       β : Type u_7
                       f : α → β
                       hf : Function.Injective f
                       x y : Finsupp β M
                       ⊢ Eq ((fun l => Finsupp.comapDomain f l ⋯) (HAdd.hAdd x y)) (HAdd.hAdd ((fun l …
                     -/
  map_add' x y := by ext; simp
                          /-
                            🎉 no goals
                          -/
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
                        β : Type u_7
                        f : α → β
                        hf : Function.Injective f
                        c : R
                        x : Finsupp β M
                        ⊢ Eq ({ toFun := fun l => Finsupp.comapDomain f l ⋯, map_add' := ⋯ }.toFun (HS …
                      -/
  map_smul' c x := by ext; simp
                           /-
                             🎉 no goals
                           -/


/-- `Finsupp.mapRange` as a `LinearMap`. -/
def mapRange.linearMap (f : M →ₗ[R] N) : (α →₀ M) →ₗ[R] α →₀ N :=
  { mapRange.addMonoidHom f.toAddMonoidHom with
    toFun := (mapRange f f.map_zero : (α →₀ M) → α →₀ N)
    -- Porting note: `hf` should be specified.
    map_smul' := fun c v => mapRange_smul (hf := f.map_zero) c v (f.map_smul c) }

-- Porting note: This was generated by `simps!`.

@[simp]
theorem mapRange.linearMap_apply (f : M →ₗ[R] N) (g : α →₀ M) :
    mapRange.linearMap f g = mapRange f f.map_zero g := rfl


@[simp]
theorem mapRange.linearMap_id :
    mapRange.linearMap LinearMap.id = (LinearMap.id : (α →₀ M) →ₗ[R] _) :=
  LinearMap.ext mapRange_id


theorem mapRange.linearMap_comp (f : N →ₗ[R] P) (f₂ : M →ₗ[R] N) :
    (mapRange.linearMap (f.comp f₂) : (α →₀ _) →ₗ[R] _) =
      (mapRange.linearMap f).comp (mapRange.linearMap f₂) :=
  -- Porting note: Placeholders should be filled.
  LinearMap.ext <| mapRange_comp f f.map_zero f₂ f₂.map_zero (comp f f₂).map_zero


@[simp]
theorem mapRange.linearMap_toAddMonoidHom (f : M →ₗ[R] N) :
    (mapRange.linearMap f).toAddMonoidHom =
      (mapRange.addMonoidHom f.toAddMonoidHom : (α →₀ M) →+ _) :=
  AddMonoidHom.ext fun _ => rfl


/-- `Finsupp.mapRange` as a `LinearEquiv`. -/
def mapRange.linearEquiv (e : M ≃ₗ[R] N) : (α →₀ M) ≃ₗ[R] α →₀ N :=
  { mapRange.linearMap e.toLinearMap,
    mapRange.addEquiv e.toAddEquiv with
    toFun := mapRange e e.map_zero
    invFun := mapRange e.symm e.symm.map_zero }

-- Porting note: This was generated by `simps`.

@[simp]
theorem mapRange.linearEquiv_apply (e : M ≃ₗ[R] N) (g : α →₀ M) :
    mapRange.linearEquiv e g = mapRange.linearMap e.toLinearMap g := rfl


@[simp]
theorem mapRange.linearEquiv_refl :
    mapRange.linearEquiv (LinearEquiv.refl R M) = LinearEquiv.refl R (α →₀ M) :=
  LinearEquiv.ext mapRange_id


theorem mapRange.linearEquiv_trans (f : M ≃ₗ[R] N) (f₂ : N ≃ₗ[R] P) :
    (mapRange.linearEquiv (f.trans f₂) : (α →₀ _) ≃ₗ[R] _) =
      (mapRange.linearEquiv f).trans (mapRange.linearEquiv f₂) :=
  -- Porting note: Placeholders should be filled.
  LinearEquiv.ext <| mapRange_comp f₂ f₂.map_zero f f.map_zero (f.trans f₂).map_zero


@[simp]
theorem mapRange.linearEquiv_symm (f : M ≃ₗ[R] N) :
    ((mapRange.linearEquiv f).symm : (α →₀ _) ≃ₗ[R] _) = mapRange.linearEquiv f.symm :=
  LinearEquiv.ext fun _x => rfl

-- Porting note: This priority should be higher than `LinearEquiv.coe_toAddEquiv`.

@[simp 1500]
theorem mapRange.linearEquiv_toAddEquiv (f : M ≃ₗ[R] N) :
    (mapRange.linearEquiv f).toAddEquiv = (mapRange.addEquiv f.toAddEquiv : (α →₀ M) ≃+ _) :=
  AddEquiv.ext fun _ => rfl


@[simp]
theorem mapRange.linearEquiv_toLinearMap (f : M ≃ₗ[R] N) :
    (mapRange.linearEquiv f).toLinearMap = (mapRange.linearMap f.toLinearMap : (α →₀ M) →ₗ[R] _) :=
  LinearMap.ext fun _ => rfl


/-- The linear equivalence between `α × β →₀ M` and `α →₀ β →₀ M`.

This is the `LinearEquiv` version of `Finsupp.finsuppProdEquiv`. -/
noncomputable def finsuppProdLEquiv {α β : Type*} (R : Type*) {M : Type*} [Semiring R]
    [AddCommMonoid M] [Module R M] : (α × β →₀ M) ≃ₗ[R] α →₀ β →₀ M :=
  { finsuppProdEquiv with
    map_add' := fun f g => by
      /-
        α✝ : Type u_1
        M✝ : Type u_2
        N : Type u_3
        P : Type u_4
        R✝ : Type u_5
        S : Type u_6
        inst✝¹⁰ : Semiring R✝
        inst✝⁹ : Semiring S
        inst✝⁸ : AddCommMonoid M✝
        inst✝⁷ : Module R✝ M✝
        inst✝⁶ : AddCommMonoid N
        inst✝⁵ : Module R✝ N
        inst✝⁴ : AddCommMonoid P
        inst✝³ : Module R✝ P
        α : Type u_7
        β : Type u_8
        R : Type u_9
        M : Type u_10
        inst✝² : Semiring R
        inst✝¹ : AddCommMonoid M
        inst✝ : Module R M
        f g : Finsupp (Prod α β) M
        ⊢ Eq (__src✝.toFun (HAdd.hAdd f g)) (HAdd.hAdd (__src✝.toFun f) (__src✝.toFun  …
      -/
      ext
      /-
        case h.h
        α✝ : Type u_1
        M✝ : Type u_2
        N : Type u_3
        P : Type u_4
        R✝ : Type u_5
        S : Type u_6
        inst✝¹⁰ : Semiring R✝
        inst✝⁹ : Semiring S
        inst✝⁸ : AddCommMonoid M✝
        inst✝⁷ : Module R✝ M✝
        inst✝⁶ : AddCommMonoid N
        inst✝⁵ : Module R✝ N
        inst✝⁴ : AddCommMonoid P
        inst✝³ : Module R✝ P
        α : Type u_7
        β : Type u_8
        R : Type u_9
        M : Type u_10
        inst✝² : Semiring R
        inst✝¹ : AddCommMonoid M
        inst✝ : Module R M
        f g : Finsupp (Prod α β) M
        a✝¹ : α
        a✝ : β
        ⊢ Eq (((__src✝.toFun (HAdd.hAdd f g)) a✝¹) a✝) (((HAdd.hAdd (__src✝.toFun f) ( …
      -/
      simp [finsuppProdEquiv, curry_apply]
      /-
        🎉 no goals
      -/
    map_smul' := fun c f => by
      /-
        α✝ : Type u_1
        M✝ : Type u_2
        N : Type u_3
        P : Type u_4
        R✝ : Type u_5
        S : Type u_6
        inst✝¹⁰ : Semiring R✝
        inst✝⁹ : Semiring S
        inst✝⁸ : AddCommMonoid M✝
        inst✝⁷ : Module R✝ M✝
        inst✝⁶ : AddCommMonoid N
        inst✝⁵ : Module R✝ N
        inst✝⁴ : AddCommMonoid P
        inst✝³ : Module R✝ P
        α : Type u_7
        β : Type u_8
        R : Type u_9
        M : Type u_10
        inst✝² : Semiring R
        inst✝¹ : AddCommMonoid M
        inst✝ : Module R M
        c : R
        f : Finsupp (Prod α β) M
        ⊢ Eq ({ toFun := __src✝.toFun, map_add' := ⋯ }.toFun (HSMul.hSMul c f)) (HSMul …
      -/
      ext
      /-
        case h.h
        α✝ : Type u_1
        M✝ : Type u_2
        N : Type u_3
        P : Type u_4
        R✝ : Type u_5
        S : Type u_6
        inst✝¹⁰ : Semiring R✝
        inst✝⁹ : Semiring S
        inst✝⁸ : AddCommMonoid M✝
        inst✝⁷ : Module R✝ M✝
        inst✝⁶ : AddCommMonoid N
        inst✝⁵ : Module R✝ N
        inst✝⁴ : AddCommMonoid P
        inst✝³ : Module R✝ P
        α : Type u_7
        β : Type u_8
        R : Type u_9
        M : Type u_10
        inst✝² : Semiring R
        inst✝¹ : AddCommMonoid M
        inst✝ : Module R M
        c : R
        f : Finsupp (Prod α β) M
        a✝¹ : α
        a✝ : β
        ⊢ Eq ((({ toFun := __src✝.toFun, map_add' := ⋯ }.toFun (HSMul.hSMul c f)) a✝¹) …
      -/
      simp [finsuppProdEquiv, curry_apply] }
      /-
        🎉 no goals
      -/


@[simp]
theorem finsuppProdLEquiv_apply {α β R M : Type*} [Semiring R] [AddCommMonoid M] [Module R M]
    (f : α × β →₀ M) (x y) : finsuppProdLEquiv R f x y = f (x, y) := by
  /-
    α : Type u_7
    β : Type u_8
    R : Type u_9
    M : Type u_10
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    f : Finsupp (Prod α β) M
    x : α
    y : β
    ⊢ Eq ((((Finsupp.finsuppProdLEquiv R) f) x) y) (f { fst := x, snd := y })
  -/
  rw [finsuppProdLEquiv, LinearEquiv.coe_mk, finsuppProdEquiv, Finsupp.curry_apply]
  /-
    🎉 no goals
  -/


@[simp]
theorem finsuppProdLEquiv_symm_apply {α β R M : Type*} [Semiring R] [AddCommMonoid M] [Module R M]
    (f : α →₀ β →₀ M) (xy) : (finsuppProdLEquiv R).symm f xy = f xy.1 xy.2 := by
  conv_rhs =>
    rw [← (finsuppProdLEquiv R).apply_symm_apply f, finsuppProdLEquiv_apply]


/-- If `Subsingleton R`, then `M ≃ₗ[R] ι →₀ R` for any type `ι`. -/
@[simps]
def Module.subsingletonEquiv (R M ι : Type*) [Semiring R] [Subsingleton R] [AddCommMonoid M]
    [Module R M] : M ≃ₗ[R] ι →₀ R where
  toFun _ := 0
  invFun _ := 0
  left_inv m := by
    /-
      R✝ : Type u_1
      M✝ : Type u_2
      N : Type u_3
      inst✝⁸ : Semiring R✝
      inst✝⁷ : AddCommMonoid M✝
      inst✝⁶ : Module R✝ M✝
      inst✝⁵ : AddCommMonoid N
      inst✝⁴ : Module R✝ N
      R : Type u_4
      M : Type u_5
      ι : Type u_6
      inst✝³ : Semiring R
      inst✝² : Subsingleton R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      m : M
      ⊢ Eq ((fun x => 0) ({ toFun := fun x => 0, map_add' := ⋯, map_smul' := ⋯ }.toF …
    -/
    letI := Module.subsingleton R M
    /-
      R✝ : Type u_1
      M✝ : Type u_2
      N : Type u_3
      inst✝⁸ : Semiring R✝
      inst✝⁷ : AddCommMonoid M✝
      inst✝⁶ : Module R✝ M✝
      inst✝⁵ : AddCommMonoid N
      inst✝⁴ : Module R✝ N
      R : Type u_4
      M : Type u_5
      ι : Type u_6
      inst✝³ : Semiring R
      inst✝² : Subsingleton R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      m : M
      this : Subsingleton M := Module.subsingleton R M
      ⊢ Eq ((fun x => 0) ({ toFun := fun x => 0, map_add' := ⋯, map_smul' := ⋯ }.toF …
    -/
    simp only [eq_iff_true_of_subsingleton]
    /-
      🎉 no goals
    -/
                    /-
                      R✝ : Type u_1
                      M✝ : Type u_2
                      N : Type u_3
                      inst✝⁸ : Semiring R✝
                      inst✝⁷ : AddCommMonoid M✝
                      inst✝⁶ : Module R✝ M✝
                      inst✝⁵ : AddCommMonoid N
                      inst✝⁴ : Module R✝ N
                      R : Type u_4
                      M : Type u_5
                      ι : Type u_6
                      inst✝³ : Semiring R
                      inst✝² : Subsingleton R
                      inst✝¹ : AddCommMonoid M
                      inst✝ : Module R M
                      f : Finsupp ι R
                      ⊢ Eq ({ toFun := fun x => 0, map_add' := ⋯, map_smul' := ⋯ }.toFun ((fun x =>  …
                    -/
  right_inv f := by simp only [eq_iff_true_of_subsingleton]
                    /-
                      🎉 no goals
                    -/
  map_add' _ _ := (add_zero 0).symm
  map_smul' r _ := (smul_zero r).symm


