local notation M " ◁ " f => fun i j h ↦ LinearMap.lTensor M (f _ _ h)

local notation f " ▷ " N => fun i j h ↦ LinearMap.rTensor N (f _ _ h)


/--
the map `limᵢ (Gᵢ ⊗ M) → (limᵢ Gᵢ) ⊗ M` induced by the family of maps `Gᵢ ⊗ M → (limᵢ Gᵢ) ⊗ M`
given by `gᵢ ⊗ m ↦ [gᵢ] ⊗ m`.
-/
noncomputable def fromDirectLimit :
    DirectLimit (G · ⊗[R] M) (f ▷ M) →ₗ[R] DirectLimit G f ⊗[R] M :=
  Module.DirectLimit.lift _ _ _ _ (fun _ ↦ (of _ _ _ _ _).rTensor M)
                     /-
                       R : Type u_1
                       inst✝⁶ : CommSemiring R
                       ι : Type u_2
                       inst✝⁵ : DecidableEq ι
                       inst✝⁴ : Preorder ι
                       G : ι → Type u_3
                       inst✝³ : (i : ι) → AddCommMonoid (G i)
                       inst✝² : (i : ι) → Module R (G i)
                       f : (i j : ι) → LE.le i j → LinearMap (RingHom.id R) (G i) (G j)
                       M : Type u_4
                       inst✝¹ : AddCommMonoid M
                       inst✝ : Module R M
                       x✝² x✝¹ : ι
                       x✝ : LE.le x✝² x✝¹
                       x : TensorProduct R (G x✝²) M
                       ⊢ Eq (((fun x => LinearMap.rTensor M (Module.DirectLimit.of R ι G f x)) x✝¹) ( …
                     -/
                                                        /-
                                                          🎉 no goals
                                                        -/
                                                        /-
                                                          🎉 no goals
                                                        -/
    fun _ _ _ x ↦ by refine x.induction_on ?_ ?_ ?_ <;> aesop
                                                        /-
                                                          🎉 no goals
                                                        -/


variable {M} in
@[simp] lemma fromDirectLimit_of_tmul {i : ι} (g : G i) (m : M) :
    fromDirectLimit f M (of _ _ _ _ i (g ⊗ₜ m)) = (of _ _ _ f i g) ⊗ₜ m :=
  lift_of (G := (G · ⊗[R] M)) _ _ (g ⊗ₜ m)


/--
the map `(limᵢ Gᵢ) ⊗ M → limᵢ (Gᵢ ⊗ M)` from the bilinear map `limᵢ Gᵢ → M → limᵢ (Gᵢ ⊗ M)` given
by the family of maps `Gᵢ → M → limᵢ (Gᵢ ⊗ M)` where `gᵢ ↦ m ↦ [gᵢ ⊗ m]`

-/
noncomputable def toDirectLimit : DirectLimit G f ⊗[R] M →ₗ[R] DirectLimit (G · ⊗[R] M) (f ▷ M) :=
  TensorProduct.lift <| Module.DirectLimit.lift _ _ _ _
    (fun i ↦
      (TensorProduct.mk R _ _).compr₂ (of R ι _ (fun _i _j h ↦ (f _ _ h).rTensor M) i))
    fun _ _ _ g ↦ DFunLike.ext _ _ (of_f (G := (G · ⊗[R] M)) (x := g ⊗ₜ ·))


variable {M} in
@[simp] lemma toDirectLimit_tmul_of
    {i : ι} (g : G i) (m : M) :
    (toDirectLimit f M <| (of _ _ G f i g) ⊗ₜ m) = (of _ _ _ _ i (g ⊗ₜ m)) := by
  /-
    R : Type u_1
    inst✝⁶ : CommSemiring R
    ι : Type u_2
    inst✝⁵ : DecidableEq ι
    inst✝⁴ : Preorder ι
    G : ι → Type u_3
    inst✝³ : (i : ι) → AddCommMonoid (G i)
    inst✝² : (i : ι) → Module R (G i)
    f : (i j : ι) → LE.le i j → LinearMap (RingHom.id R) (G i) (G j)
    M : Type u_4
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    i : ι
    g : G i
    m : M
    ⊢ Eq ((TensorProduct.toDirectLimit f M) (TensorProduct.tmul R ((Module.DirectL …
  -/
  rw [toDirectLimit, lift.tmul, lift_of]
  /-
    R : Type u_1
    inst✝⁶ : CommSemiring R
    ι : Type u_2
    inst✝⁵ : DecidableEq ι
    inst✝⁴ : Preorder ι
    G : ι → Type u_3
    inst✝³ : (i : ι) → AddCommMonoid (G i)
    inst✝² : (i : ι) → Module R (G i)
    f : (i j : ι) → LE.le i j → LinearMap (RingHom.id R) (G i) (G j)
    M : Type u_4
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    i : ι
    g : G i
    m : M
    ⊢ Eq ((((TensorProduct.mk R (G i) M).compr₂ (Module.DirectLimit.of R ι (fun _i …
  -/
  rfl
  /-
    🎉 no goals
  -/


/--
`limᵢ (Gᵢ ⊗ M)` and `(limᵢ Gᵢ) ⊗ M` are isomorphic as modules
-/
noncomputable def directLimitLeft :
    DirectLimit G f ⊗[R] M ≃ₗ[R] DirectLimit (G · ⊗[R] M) (f ▷ M) := by
  /-
    R : Type u_1
    inst✝⁶ : CommSemiring R
    ι : Type u_2
    inst✝⁵ : DecidableEq ι
    inst✝⁴ : Preorder ι
    G : ι → Type u_3
    inst✝³ : (i : ι) → AddCommMonoid (G i)
    inst✝² : (i : ι) → Module R (G i)
    f : (i j : ι) → LE.le i j → LinearMap (RingHom.id R) (G i) (G j)
    M : Type u_4
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    ⊢ LinearEquiv (RingHom.id R) (TensorProduct R (Module.DirectLimit G f) M) (Mod …
  -/
  refine LinearEquiv.ofLinear (toDirectLimit f M) (fromDirectLimit f M) ?_ (ext ?_)
    /-
      case refine_1
      R : Type u_1
      inst✝⁶ : CommSemiring R
      ι : Type u_2
      inst✝⁵ : DecidableEq ι
      inst✝⁴ : Preorder ι
      G : ι → Type u_3
      inst✝³ : (i : ι) → AddCommMonoid (G i)
      inst✝² : (i : ι) → Module R (G i)
      f : (i j : ι) → LE.le i j → LinearMap (RingHom.id R) (G i) (G j)
      M : Type u_4
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      ⊢ Eq ((TensorProduct.toDirectLimit f M).comp (TensorProduct.fromDirectLimit f  …
    -/
  · ext ⟨x⟩
    exact x.induction_on (by simp) (fun i x ↦ x.induction_on (by simp)
      (fun _ _ ↦ by rw [quotMk_of]; simp) <| by simp+contextual) (by simp+contextual)
    /-
      case refine_2
      R : Type u_1
      inst✝⁶ : CommSemiring R
      ι : Type u_2
      inst✝⁵ : DecidableEq ι
      inst✝⁴ : Preorder ι
      G : ι → Type u_3
      inst✝³ : (i : ι) → AddCommMonoid (G i)
      inst✝² : (i : ι) → Module R (G i)
      f : (i j : ι) → LE.le i j → LinearMap (RingHom.id R) (G i) (G j)
      M : Type u_4
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      ⊢ Eq ((TensorProduct.mk R (Module.DirectLimit G f) M).compr₂ ((TensorProduct.f …
    -/
  · ext ⟨x⟩ m
    /-
      case refine_2.h.mk.h
      R : Type u_1
      inst✝⁶ : CommSemiring R
      ι : Type u_2
      inst✝⁵ : DecidableEq ι
      inst✝⁴ : Preorder ι
      G : ι → Type u_3
      inst✝³ : (i : ι) → AddCommMonoid (G i)
      inst✝² : (i : ι) → Module R (G i)
      f : (i j : ι) → LE.le i j → LinearMap (RingHom.id R) (G i) (G j)
      M : Type u_4
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      x✝ : Module.DirectLimit G f
      x : DirectSum ι G
      m : M
      ⊢ Eq ((((TensorProduct.mk R (Module.DirectLimit G f) M).compr₂ ((TensorProduct …
    -/
    exact x.induction_on (by simp) (fun _ _ ↦ by rw [quotMk_of]; simp) (by simp+contextual)
    /-
      🎉 no goals
    -/


@[simp] lemma directLimitLeft_tmul_of {i : ι} (g : G i) (m : M) :
    directLimitLeft f M (of _ _ _ _ _ g ⊗ₜ m) = of _ _ _ (f ▷ M) _ (g ⊗ₜ m) :=
  toDirectLimit_tmul_of f g m


@[simp] lemma directLimitLeft_symm_of_tmul {i : ι} (g : G i) (m : M) :
    (directLimitLeft f M).symm (of _ _ _ _ _ (g ⊗ₜ m)) = of _ _ _ f _ g ⊗ₜ m :=
  fromDirectLimit_of_tmul f g m


lemma directLimitLeft_rTensor_of {i : ι} (x : G i ⊗[R] M) :
    directLimitLeft f M (LinearMap.rTensor M (of ..) x) = of _ _ _ (f ▷ M) _ x :=
                     /-
                       R : Type u_1
                       inst✝⁶ : CommSemiring R
                       ι : Type u_2
                       inst✝⁵ : DecidableEq ι
                       inst✝⁴ : Preorder ι
                       G : ι → Type u_3
                       inst✝³ : (i : ι) → AddCommMonoid (G i)
                       inst✝² : (i : ι) → Module R (G i)
                       f : (i j : ι) → LE.le i j → LinearMap (RingHom.id R) (G i) (G j)
                       M : Type u_4
                       inst✝¹ : AddCommMonoid M
                       inst✝ : Module R M
                       i : ι
                       x : TensorProduct R (G i) M
                       ⊢ Eq ((TensorProduct.directLimitLeft f M) ((LinearMap.rTensor M (Module.Direct …
                     -/
                     /-
                       🎉 no goals
                     -/
                               /-
                                 🎉 no goals
                               -/
  x.induction_on (by simp) (by simp+contextual) (by simp+contextual)
                                                    /-
                                                      🎉 no goals
                                                    -/


/--
`M ⊗ (limᵢ Gᵢ)` and `limᵢ (M ⊗ Gᵢ)` are isomorphic as modules
-/
noncomputable def directLimitRight :
    M ⊗[R] DirectLimit G f ≃ₗ[R] DirectLimit (M ⊗[R] G ·) (M ◁ f) :=
  TensorProduct.comm _ _ _ ≪≫ₗ directLimitLeft f M ≪≫ₗ
    Module.DirectLimit.congr (fun _ ↦ TensorProduct.comm _ _ _)
                                                               /-
                                                                 R : Type u_1
                                                                 inst✝⁶ : CommSemiring R
                                                                 ι : Type u_2
                                                                 inst✝⁵ : DecidableEq ι
                                                                 inst✝⁴ : Preorder ι
                                                                 G : ι → Type u_3
                                                                 inst✝³ : (i : ι) → AddCommMonoid (G i)
                                                                 inst✝² : (i : ι) → Module R (G i)
                                                                 f : (i j : ι) → LE.le i j → LinearMap (RingHom.id R) (G i) (G j)
                                                                 M : Type u_4
                                                                 inst✝¹ : AddCommMonoid M
                                                                 inst✝ : Module R M
                                                                 i j : ι
                                                                 h : LE.le i j
                                                                 ⊢ ∀ (x : G i), Eq (((TensorProduct.mk R (G i) M).compr₂ ((↑((fun x => TensorPr …
                                                               -/
      (fun i j h ↦ TensorProduct.ext <| DFunLike.ext _ _ <| by aesop)
                                                               /-
                                                                 🎉 no goals
                                                               -/


@[simp] lemma directLimitRight_tmul_of {i : ι} (m : M) (g : G i) :
    directLimitRight f M (m ⊗ₜ of _ _ _ _ _ g) = of _ _ _ _ i (m ⊗ₜ g) := by
  /-
    R : Type u_1
    inst✝⁶ : CommSemiring R
    ι : Type u_2
    inst✝⁵ : DecidableEq ι
    inst✝⁴ : Preorder ι
    G : ι → Type u_3
    inst✝³ : (i : ι) → AddCommMonoid (G i)
    inst✝² : (i : ι) → Module R (G i)
    f : (i j : ι) → LE.le i j → LinearMap (RingHom.id R) (G i) (G j)
    M : Type u_4
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    i : ι
    m : M
    g : G i
    ⊢ Eq ((TensorProduct.directLimitRight f M) (TensorProduct.tmul R m ((Module.Di …
  -/
  simp [directLimitRight, congr_apply_of]
  /-
    🎉 no goals
  -/


@[simp] lemma directLimitRight_symm_of_tmul {i : ι} (m : M) (g : G i) :
    (directLimitRight f M).symm (of _ _ _ _ _ (m ⊗ₜ g)) = m ⊗ₜ of _ _ _ f _ g := by
  /-
    R : Type u_1
    inst✝⁶ : CommSemiring R
    ι : Type u_2
    inst✝⁵ : DecidableEq ι
    inst✝⁴ : Preorder ι
    G : ι → Type u_3
    inst✝³ : (i : ι) → AddCommMonoid (G i)
    inst✝² : (i : ι) → Module R (G i)
    f : (i j : ι) → LE.le i j → LinearMap (RingHom.id R) (G i) (G j)
    M : Type u_4
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    i : ι
    m : M
    g : G i
    ⊢ Eq ((TensorProduct.directLimitRight f M).symm ((Module.DirectLimit.of R ι (f …
  -/
  simp [directLimitRight, congr_symm_apply_of]
  /-
    🎉 no goals
  -/


instance : DirectedSystem (G · ⊗[R] M) (f ▷ M) where
  map_self i x := by
    /-
      R : Type u_1
      inst✝⁷ : CommSemiring R
      ι : Type u_2
      inst✝⁶ : DecidableEq ι
      inst✝⁵ : Preorder ι
      G : ι → Type u_3
      inst✝⁴ : (i : ι) → AddCommMonoid (G i)
      inst✝³ : (i : ι) → Module R (G i)
      f : (i j : ι) → LE.le i j → LinearMap (RingHom.id R) (G i) (G j)
      M : Type u_4
      inst✝² : AddCommMonoid M
      inst✝¹ : Module R M
      inst✝ : DirectedSystem G fun x1 x2 x3 => ⇑(f x1 x2 x3)
      i : ι
      x : TensorProduct R (G i) M
      ⊢ Eq ((LinearMap.rTensor M (f i i ⋯)) x) x
    -/
    convert LinearMap.rTensor_id_apply M (G i) x; ext; apply DirectedSystem.map_self'
                                                       /-
                                                         🎉 no goals
                                                       -/
  map_map _ _ _ _ _ x := by
    /-
      R : Type u_1
      inst✝⁷ : CommSemiring R
      ι : Type u_2
      inst✝⁶ : DecidableEq ι
      inst✝⁵ : Preorder ι
      G : ι → Type u_3
      inst✝⁴ : (i : ι) → AddCommMonoid (G i)
      inst✝³ : (i : ι) → Module R (G i)
      f : (i j : ι) → LE.le i j → LinearMap (RingHom.id R) (G i) (G j)
      M : Type u_4
      inst✝² : AddCommMonoid M
      inst✝¹ : Module R M
      inst✝ : DirectedSystem G fun x1 x2 x3 => ⇑(f x1 x2 x3)
      x✝⁴ x✝³ x✝² : ι
      x✝¹ : LE.le x✝² x✝³
      x✝ : LE.le x✝³ x✝⁴
      x : TensorProduct R (G x✝²) M
      ⊢ Eq ((LinearMap.rTensor M (f x✝³ x✝⁴ x✝)) ((LinearMap.rTensor M (f x✝² x✝³ x✝ …
    -/
    convert ← (LinearMap.rTensor_comp_apply M _ _ x).symm; ext; apply DirectedSystem.map_map' f
                                                                /-
                                                                  🎉 no goals
                                                                -/


instance : DirectedSystem (M ⊗[R] G ·) (M ◁ f) where
  map_self i x := by
    /-
      R : Type u_1
      inst✝⁷ : CommSemiring R
      ι : Type u_2
      inst✝⁶ : DecidableEq ι
      inst✝⁵ : Preorder ι
      G : ι → Type u_3
      inst✝⁴ : (i : ι) → AddCommMonoid (G i)
      inst✝³ : (i : ι) → Module R (G i)
      f : (i j : ι) → LE.le i j → LinearMap (RingHom.id R) (G i) (G j)
      M : Type u_4
      inst✝² : AddCommMonoid M
      inst✝¹ : Module R M
      inst✝ : DirectedSystem G fun x1 x2 x3 => ⇑(f x1 x2 x3)
      i : ι
      x : TensorProduct R M (G i)
      ⊢ Eq ((LinearMap.lTensor M (f i i ⋯)) x) x
    -/
    convert LinearMap.lTensor_id_apply M _ x; ext; apply DirectedSystem.map_self'
                                                   /-
                                                     🎉 no goals
                                                   -/
  map_map _ _ _ h₁ h₂ x := by
    /-
      R : Type u_1
      inst✝⁷ : CommSemiring R
      ι : Type u_2
      inst✝⁶ : DecidableEq ι
      inst✝⁵ : Preorder ι
      G : ι → Type u_3
      inst✝⁴ : (i : ι) → AddCommMonoid (G i)
      inst✝³ : (i : ι) → Module R (G i)
      f : (i j : ι) → LE.le i j → LinearMap (RingHom.id R) (G i) (G j)
      M : Type u_4
      inst✝² : AddCommMonoid M
      inst✝¹ : Module R M
      inst✝ : DirectedSystem G fun x1 x2 x3 => ⇑(f x1 x2 x3)
      x✝² x✝¹ x✝ : ι
      h₁ : LE.le x✝ x✝¹
      h₂ : LE.le x✝¹ x✝²
      x : TensorProduct R M (G x✝)
      ⊢ Eq ((LinearMap.lTensor M (f x✝¹ x✝² h₂)) ((LinearMap.lTensor M (f x✝ x✝¹ h₁) …
    -/
    convert ← (LinearMap.lTensor_comp_apply M _ _ x).symm; ext; apply DirectedSystem.map_map' f
                                                                /-
                                                                  🎉 no goals
                                                                -/


