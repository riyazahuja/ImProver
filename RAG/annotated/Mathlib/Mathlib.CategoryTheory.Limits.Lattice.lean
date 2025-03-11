/-- The limit cone over any functor from a finite diagram into a `SemilatticeInf` with `OrderTop`.
-/
def finiteLimitCone [SemilatticeInf α] [OrderTop α] (F : J ⥤ α) : LimitCone F where
  cone :=
    { pt := Finset.univ.inf F.obj
      π := { app := fun _ => homOfLE (Finset.inf_le (Fintype.complete _)) } }
  isLimit := { lift := fun s => homOfLE (Finset.le_inf fun j _ => (s.π.app j).down.down) }


/--
The colimit cocone over any functor from a finite diagram into a `SemilatticeSup` with `OrderBot`.
-/
def finiteColimitCocone [SemilatticeSup α] [OrderBot α] (F : J ⥤ α) : ColimitCocone F where
  cocone :=
    { pt := Finset.univ.sup F.obj
      ι := { app := fun _ => homOfLE (Finset.le_sup (Fintype.complete _)) } }
  isColimit := { desc := fun s => homOfLE (Finset.sup_le fun j _ => (s.ι.app j).down.down) }

-- see Note [lower instance priority]

instance (priority := 100) hasFiniteLimits_of_semilatticeInf_orderTop [SemilatticeInf α]
    [OrderTop α] : HasFiniteLimits α := ⟨by
  /-
    α : Type u
    J : Type w
    inst✝³ : CategoryTheory.SmallCategory J
    inst✝² : CategoryTheory.FinCategory J
    inst✝¹ : SemilatticeInf α
    inst✝ : OrderTop α
    ⊢ ∀ (J : Type) [𝒥 : CategoryTheory.SmallCategory J] [inst : CategoryTheory.Fin …
  -/
  intro J 𝒥₁ 𝒥₂
  /-
    α : Type u
    J✝ : Type w
    inst✝³ : CategoryTheory.SmallCategory J✝
    inst✝² : CategoryTheory.FinCategory J✝
    inst✝¹ : SemilatticeInf α
    inst✝ : OrderTop α
    J : Type
    𝒥₁ : CategoryTheory.SmallCategory J
    𝒥₂ : CategoryTheory.FinCategory J
    ⊢ CategoryTheory.Limits.HasLimitsOfShape J α
  -/
  exact { has_limit := fun F => HasLimit.mk (finiteLimitCone F) }⟩
  /-
    🎉 no goals
  -/

-- see Note [lower instance priority]

instance (priority := 100) hasFiniteColimits_of_semilatticeSup_orderBot [SemilatticeSup α]
    [OrderBot α] : HasFiniteColimits α := ⟨by
  /-
    α : Type u
    J : Type w
    inst✝³ : CategoryTheory.SmallCategory J
    inst✝² : CategoryTheory.FinCategory J
    inst✝¹ : SemilatticeSup α
    inst✝ : OrderBot α
    ⊢ ∀ (J : Type) [𝒥 : CategoryTheory.SmallCategory J] [inst : CategoryTheory.Fin …
  -/
  intro J 𝒥₁ 𝒥₂
  /-
    α : Type u
    J✝ : Type w
    inst✝³ : CategoryTheory.SmallCategory J✝
    inst✝² : CategoryTheory.FinCategory J✝
    inst✝¹ : SemilatticeSup α
    inst✝ : OrderBot α
    J : Type
    𝒥₁ : CategoryTheory.SmallCategory J
    𝒥₂ : CategoryTheory.FinCategory J
    ⊢ CategoryTheory.Limits.HasColimitsOfShape J α
  -/
  exact { has_colimit := fun F => HasColimit.mk (finiteColimitCocone F) }⟩
  /-
    🎉 no goals
  -/


/-- The limit of a functor from a finite diagram into a `SemilatticeInf` with `OrderTop` is the
infimum of the objects in the image.
-/
theorem finite_limit_eq_finset_univ_inf [SemilatticeInf α] [OrderTop α] (F : J ⥤ α) :
    limit F = Finset.univ.inf F.obj :=
  (IsLimit.conePointUniqueUpToIso (limit.isLimit F) (finiteLimitCone F).isLimit).to_eq


/-- The colimit of a functor from a finite diagram into a `SemilatticeSup` with `OrderBot`
is the supremum of the objects in the image.
-/
theorem finite_colimit_eq_finset_univ_sup [SemilatticeSup α] [OrderBot α] (F : J ⥤ α) :
    colimit F = Finset.univ.sup F.obj :=
  (IsColimit.coconePointUniqueUpToIso (colimit.isColimit F) (finiteColimitCocone F).isColimit).to_eq


/--
A finite product in the category of a `SemilatticeInf` with `OrderTop` is the same as the infimum.
-/
theorem finite_product_eq_finset_inf [SemilatticeInf α] [OrderTop α] {ι : Type u} [Fintype ι]
    (f : ι → α) : ∏ᶜ f = Fintype.elems.inf f := by
  /-
    α : Type u
    inst✝² : SemilatticeInf α
    inst✝¹ : OrderTop α
    ι : Type u
    inst✝ : Fintype ι
    f : ι → α
    ⊢ Eq (CategoryTheory.Limits.piObj f) (Fintype.elems.inf f)
  -/
  trans
  · exact
      (IsLimit.conePointUniqueUpToIso (limit.isLimit _)
          (finiteLimitCone (Discrete.functor f)).isLimit).to_eq
  /-
    α : Type u
    inst✝² : SemilatticeInf α
    inst✝¹ : OrderTop α
    ι : Type u
    inst✝ : Fintype ι
    f : ι → α
    ⊢ Eq (CategoryTheory.Limits.CompleteLattice.finiteLimitCone (CategoryTheory.Di …
  -/
  change Finset.univ.inf (f ∘ discreteEquiv.toEmbedding) = Fintype.elems.inf f
  /-
    α : Type u
    inst✝² : SemilatticeInf α
    inst✝¹ : OrderTop α
    ι : Type u
    inst✝ : Fintype ι
    f : ι → α
    ⊢ Eq (Finset.univ.inf (Function.comp f ⇑CategoryTheory.discreteEquiv.toEmbeddi …
  -/
  simp only [← Finset.inf_map, Finset.univ_map_equiv_to_embedding]
  /-
    α : Type u
    inst✝² : SemilatticeInf α
    inst✝¹ : OrderTop α
    ι : Type u
    inst✝ : Fintype ι
    f : ι → α
    ⊢ Eq (Finset.univ.inf f) (Fintype.elems.inf f)
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- A finite coproduct in the category of a `SemilatticeSup` with `OrderBot` is the same as the
supremum.
-/
theorem finite_coproduct_eq_finset_sup [SemilatticeSup α] [OrderBot α] {ι : Type u} [Fintype ι]
    (f : ι → α) : ∐ f = Fintype.elems.sup f := by
  /-
    α : Type u
    inst✝² : SemilatticeSup α
    inst✝¹ : OrderBot α
    ι : Type u
    inst✝ : Fintype ι
    f : ι → α
    ⊢ Eq (CategoryTheory.Limits.sigmaObj f) (Fintype.elems.sup f)
  -/
  trans
  · exact
      (IsColimit.coconePointUniqueUpToIso (colimit.isColimit _)
          (finiteColimitCocone (Discrete.functor f)).isColimit).to_eq
  /-
    α : Type u
    inst✝² : SemilatticeSup α
    inst✝¹ : OrderBot α
    ι : Type u
    inst✝ : Fintype ι
    f : ι → α
    ⊢ Eq (CategoryTheory.Limits.CompleteLattice.finiteColimitCocone (CategoryTheor …
  -/
  change Finset.univ.sup (f ∘ discreteEquiv.toEmbedding) = Fintype.elems.sup f
  /-
    α : Type u
    inst✝² : SemilatticeSup α
    inst✝¹ : OrderBot α
    ι : Type u
    inst✝ : Fintype ι
    f : ι → α
    ⊢ Eq (Finset.univ.sup (Function.comp f ⇑CategoryTheory.discreteEquiv.toEmbeddi …
  -/
  simp only [← Finset.sup_map, Finset.univ_map_equiv_to_embedding]
  /-
    α : Type u
    inst✝² : SemilatticeSup α
    inst✝¹ : OrderBot α
    ι : Type u
    inst✝ : Fintype ι
    f : ι → α
    ⊢ Eq (Finset.univ.sup f) (Fintype.elems.sup f)
  -/
  rfl
  /-
    🎉 no goals
  -/

-- see Note [lower instance priority]

instance (priority := 100) [SemilatticeInf α] [OrderTop α] : HasBinaryProducts α := by
  have : ∀ x y : α, HasLimit (pair x y) := by
    letI := hasFiniteLimits_of_hasFiniteLimits_of_size.{u} α
    infer_instance
  /-
    α : Type u
    J : Type w
    inst✝³ : CategoryTheory.SmallCategory J
    inst✝² : CategoryTheory.FinCategory J
    inst✝¹ : SemilatticeInf α
    inst✝ : OrderTop α
    this : ∀ (x y : α), CategoryTheory.Limits.HasLimit (CategoryTheory.Limits.pair …
    ⊢ CategoryTheory.Limits.HasBinaryProducts α
  -/
  apply hasBinaryProducts_of_hasLimit_pair
  /-
    🎉 no goals
  -/


/-- The binary product in the category of a `SemilatticeInf` with `OrderTop` is the same as the
infimum.
-/
@[simp]
theorem prod_eq_inf [SemilatticeInf α] [OrderTop α] (x y : α) : Limits.prod x y = x ⊓ y :=
  calc
    Limits.prod x y = limit (pair x y) := rfl
                                             /-
                                               α : Type u
                                               inst✝¹ : SemilatticeInf α
                                               inst✝ : OrderTop α
                                               x y : α
                                               ⊢ Eq (CategoryTheory.Limits.limit (CategoryTheory.Limits.pair x y)) (Finset.un …
                                             -/
    _ = Finset.univ.inf (pair x y).obj := by rw [finite_limit_eq_finset_univ_inf (pair.{u} x y)]
                                             /-
                                               🎉 no goals
                                             -/
    _ = x ⊓ (y ⊓ ⊤) := rfl
    -- Note: finset.inf is realized as a fold, hence the definitional equality
                    /-
                      α : Type u
                      inst✝¹ : SemilatticeInf α
                      inst✝ : OrderTop α
                      x y : α
                      ⊢ Eq (Min.min x (Min.min y Top.top)) (Min.min x y)
                    -/
    _ = x ⊓ y := by rw [inf_top_eq]
                    /-
                      🎉 no goals
                    -/

-- see Note [lower instance priority]

instance (priority := 100) [SemilatticeSup α] [OrderBot α] : HasBinaryCoproducts α := by
  have : ∀ x y : α, HasColimit (pair x y) := by
    letI := hasFiniteColimits_of_hasFiniteColimits_of_size.{u} α
    infer_instance
  /-
    α : Type u
    J : Type w
    inst✝³ : CategoryTheory.SmallCategory J
    inst✝² : CategoryTheory.FinCategory J
    inst✝¹ : SemilatticeSup α
    inst✝ : OrderBot α
    this : ∀ (x y : α), CategoryTheory.Limits.HasColimit (CategoryTheory.Limits.pa …
    ⊢ CategoryTheory.Limits.HasBinaryCoproducts α
  -/
  apply hasBinaryCoproducts_of_hasColimit_pair
  /-
    🎉 no goals
  -/


/-- The binary coproduct in the category of a `SemilatticeSup` with `OrderBot` is the same as the
supremum.
-/
@[simp]
theorem coprod_eq_sup [SemilatticeSup α] [OrderBot α] (x y : α) : Limits.coprod x y = x ⊔ y :=
  calc
    Limits.coprod x y = colimit (pair x y) := rfl
                                             /-
                                               α : Type u
                                               inst✝¹ : SemilatticeSup α
                                               inst✝ : OrderBot α
                                               x y : α
                                               ⊢ Eq (CategoryTheory.Limits.colimit (CategoryTheory.Limits.pair x y)) (Finset. …
                                             -/
    _ = Finset.univ.sup (pair x y).obj := by rw [finite_colimit_eq_finset_univ_sup (pair x y)]
                                             /-
                                               🎉 no goals
                                             -/
    _ = x ⊔ (y ⊔ ⊥) := rfl
    -- Note: Finset.sup is realized as a fold, hence the definitional equality
                    /-
                      α : Type u
                      inst✝¹ : SemilatticeSup α
                      inst✝ : OrderBot α
                      x y : α
                      ⊢ Eq (Max.max x (Max.max y Bot.bot)) (Max.max x y)
                    -/
    _ = x ⊔ y := by rw [sup_bot_eq]
                    /-
                      🎉 no goals
                    -/


/-- The pullback in the category of a `SemilatticeInf` with `OrderTop` is the same as the infimum
over the objects.
-/
@[simp]
theorem pullback_eq_inf [SemilatticeInf α] [OrderTop α] {x y z : α} (f : x ⟶ z) (g : y ⟶ z) :
    pullback f g = x ⊓ y :=
  calc
    pullback f g = limit (cospan f g) := rfl
                                               /-
                                                 α : Type u
                                                 inst✝¹ : SemilatticeInf α
                                                 inst✝ : OrderTop α
                                                 x y z : α
                                                 f : Quiver.Hom x z
                                                 g : Quiver.Hom y z
                                                 ⊢ Eq (CategoryTheory.Limits.limit (CategoryTheory.Limits.cospan f g)) (Finset. …
                                               -/
    _ = Finset.univ.inf (cospan f g).obj := by rw [finite_limit_eq_finset_univ_inf]
                                               /-
                                                 🎉 no goals
                                               -/
    _ = z ⊓ (x ⊓ (y ⊓ ⊤)) := rfl
                          /-
                            α : Type u
                            inst✝¹ : SemilatticeInf α
                            inst✝ : OrderTop α
                            x y z : α
                            f : Quiver.Hom x z
                            g : Quiver.Hom y z
                            ⊢ Eq (Min.min z (Min.min x (Min.min y Top.top))) (Min.min z (Min.min x y))
                          -/
    _ = z ⊓ (x ⊓ y) := by rw [inf_top_eq]
                          /-
                            🎉 no goals
                          -/
    _ = x ⊓ y := inf_eq_right.mpr (inf_le_of_left_le f.le)


/-- The pushout in the category of a `SemilatticeSup` with `OrderBot` is the same as the supremum
over the objects.
-/
@[simp]
theorem pushout_eq_sup [SemilatticeSup α] [OrderBot α] (x y z : α) (f : z ⟶ x) (g : z ⟶ y) :
    pushout f g = x ⊔ y :=
  calc
    pushout f g = colimit (span f g) := rfl
                                             /-
                                               α : Type u
                                               inst✝¹ : SemilatticeSup α
                                               inst✝ : OrderBot α
                                               x y z : α
                                               f : Quiver.Hom z x
                                               g : Quiver.Hom z y
                                               ⊢ Eq (CategoryTheory.Limits.colimit (CategoryTheory.Limits.span f g)) (Finset. …
                                             -/
    _ = Finset.univ.sup (span f g).obj := by rw [finite_colimit_eq_finset_univ_sup]
                                             /-
                                               🎉 no goals
                                             -/
    _ = z ⊔ (x ⊔ (y ⊔ ⊥)) := rfl
                          /-
                            α : Type u
                            inst✝¹ : SemilatticeSup α
                            inst✝ : OrderBot α
                            x y z : α
                            f : Quiver.Hom z x
                            g : Quiver.Hom z y
                            ⊢ Eq (Max.max z (Max.max x (Max.max y Bot.bot))) (Max.max z (Max.max x y))
                          -/
    _ = z ⊔ (x ⊔ y) := by rw [sup_bot_eq]
                          /-
                            🎉 no goals
                          -/
    _ = x ⊔ y := sup_eq_right.mpr (le_sup_of_le_left f.le)


/-- The limit cone over any functor into a complete lattice.
-/
def limitCone (F : J ⥤ α) : LimitCone F where
  cone :=
    { pt := iInf F.obj
      π := { app := fun _ => homOfLE (CompleteLattice.sInf_le _ _ (Set.mem_range_self _)) } }
  isLimit :=
    { lift := fun s =>
                                                 /-
                                                   α : Type u
                                                   inst✝¹ : CompleteLattice α
                                                   J : Type u
                                                   inst✝ : CategoryTheory.SmallCategory J
                                                   F : CategoryTheory.Functor J α
                                                   s : CategoryTheory.Limits.Cone F
                                                   ⊢ ∀ (b : α), Membership.mem (Set.range F.obj) b → LE.le s.pt b
                                                 -/
        homOfLE (CompleteLattice.le_sInf _ _ (by rintro _ ⟨j, rfl⟩; exact (s.π.app j).le)) }
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


/-- The colimit cocone over any functor into a complete lattice.
-/
def colimitCocone (F : J ⥤ α) : ColimitCocone F where
  cocone :=
    { pt := iSup F.obj
      ι := { app := fun _ => homOfLE (CompleteLattice.le_sSup _ _ (Set.mem_range_self _)) } }
  isColimit :=
    { desc := fun s =>
                                                 /-
                                                   α : Type u
                                                   inst✝¹ : CompleteLattice α
                                                   J : Type u
                                                   inst✝ : CategoryTheory.SmallCategory J
                                                   F : CategoryTheory.Functor J α
                                                   s : CategoryTheory.Limits.Cocone F
                                                   ⊢ ∀ (b : α), Membership.mem (Set.range F.obj) b → LE.le b s.pt
                                                 -/
        homOfLE (CompleteLattice.sSup_le _ _ (by rintro _ ⟨j, rfl⟩; exact (s.ι.app j).le)) }
                                                                    /-
                                                                      🎉 no goals
                                                                    -/

-- It would be nice to only use the `Inf` half of the complete lattice, but
-- this seems not to have been described separately.
-- see Note [lower instance priority]

instance (priority := 100) hasLimits_of_completeLattice : HasLimits α where
  has_limits_of_shape _ := { has_limit := fun F => HasLimit.mk (limitCone F) }

-- see Note [lower instance priority]

instance (priority := 100) hasColimits_of_completeLattice : HasColimits α where
  has_colimits_of_shape _ := { has_colimit := fun F => HasColimit.mk (colimitCocone F) }


/-- The limit of a functor into a complete lattice is the infimum of the objects in the image.
-/
theorem limit_eq_iInf (F : J ⥤ α) : limit F = iInf F.obj :=
  (IsLimit.conePointUniqueUpToIso (limit.isLimit F) (limitCone F).isLimit).to_eq


/-- The colimit of a functor into a complete lattice is the supremum of the objects in the image.
-/
theorem colimit_eq_iSup (F : J ⥤ α) : colimit F = iSup F.obj :=
  (IsColimit.coconePointUniqueUpToIso (colimit.isColimit F) (colimitCocone F).isColimit).to_eq


