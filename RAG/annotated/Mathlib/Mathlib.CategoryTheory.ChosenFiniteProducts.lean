/--
An instance of `ChosenFiniteProducts C` bundles an explicit choice of a binary
product of two objects of `C`, and a terminal object in `C`.

Users should use the monoidal notation: `X ⊗ Y` for the product and `𝟙_ C` for
the terminal object.
-/
class ChosenFiniteProducts (C : Type u) [Category.{v} C] where
  /-- A choice of a limit binary fan for any two objects of the category. -/
  product : (X Y : C) → Limits.LimitCone (Limits.pair X Y)
  /-- A choice of a terminal object. -/
  terminal : Limits.LimitCone (Functor.empty.{0} C)


instance (priority := 100) (C : Type u) [Category.{v} C] [ChosenFiniteProducts C] :
    MonoidalCategory C :=
  monoidalOfChosenFiniteProducts terminal product


instance (priority := 100) (C : Type u) [Category.{v} C] [ChosenFiniteProducts C] :
    SymmetricCategory C :=
  symmetricOfChosenFiniteProducts _ _


/--
The unique map to the terminal object.
-/
def toUnit (X : C) : X ⟶ 𝟙_ C :=
  terminal.isLimit.lift <| .mk _ <| .mk (fun x => x.as.elim) fun x => x.as.elim


instance (X : C) : Unique (X ⟶ 𝟙_ C) where
  default := toUnit _
  uniq _ := terminal.isLimit.hom_ext fun ⟨j⟩ => j.elim


/--
This lemma follows from the preexisting `Unique` instance, but
it is often convenient to use it directly as `apply toUnit_unique` forcing
lean to do the necessary elaboration.
-/
lemma toUnit_unique {X : C} (f g : X ⟶ 𝟙_ _) : f = g :=
  Subsingleton.elim _ _


/--
Construct a morphism to the product given its two components.
-/
def lift {T X Y : C} (f : T ⟶ X) (g : T ⟶ Y) : T ⟶ X ⊗ Y :=
  (product X Y).isLimit.lift <| Limits.BinaryFan.mk f g


/--
The first projection from the product.
-/
def fst (X Y : C) : X ⊗ Y ⟶ X :=
  letI F : Limits.BinaryFan X Y := (product X Y).cone
  F.fst


/--
The second projection from the product.
-/
def snd (X Y : C) : X ⊗ Y ⟶ Y :=
  letI F : Limits.BinaryFan X Y := (product X Y).cone
  F.snd


@[reassoc (attr := simp)]
lemma lift_fst {T X Y : C} (f : T ⟶ X) (g : T ⟶ Y) : lift f g ≫ fst _ _ = f := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.ChosenFiniteProducts C
    T X Y : C
    f : Quiver.Hom T X
    g : Quiver.Hom T Y
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.ChosenFiniteProducts. …
  -/
  simp [lift, fst]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma lift_snd {T X Y : C} (f : T ⟶ X) (g : T ⟶ Y) : lift f g ≫ snd _ _ = g := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.ChosenFiniteProducts C
    T X Y : C
    f : Quiver.Hom T X
    g : Quiver.Hom T Y
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.ChosenFiniteProducts. …
  -/
  simp [lift, snd]
  /-
    🎉 no goals
  -/


instance mono_lift_of_mono_left {W X Y : C} (f : W ⟶ X) (g : W ⟶ Y)
    [Mono f] : Mono (lift f g) :=
  mono_of_mono_fac <| lift_fst _ _


instance mono_lift_of_mono_right {W X Y : C} (f : W ⟶ X) (g : W ⟶ Y)
    [Mono g] : Mono (lift f g) :=
  mono_of_mono_fac <| lift_snd _ _


@[ext 1050]
lemma hom_ext {T X Y : C} (f g : T ⟶ X ⊗ Y)
    (h_fst : f ≫ fst _ _ = g ≫ fst _ _)
    (h_snd : f ≫ snd _ _ = g ≫ snd _ _) :
    f = g :=
  (product X Y).isLimit.hom_ext fun ⟨j⟩ => j.recOn h_fst h_snd

-- Similarly to `CategoryTheory.Limits.prod.comp_lift`, we do not make the `assoc` version a simp
-- lemma

@[reassoc, simp]
lemma comp_lift {V W X Y : C} (f : V ⟶ W) (g : W ⟶ X) (h : W ⟶ Y) :
                                              /-
                                                C : Type u
                                                inst✝¹ : CategoryTheory.Category.{v, u} C
                                                inst✝ : CategoryTheory.ChosenFiniteProducts C
                                                V W X Y : C
                                                f : Quiver.Hom V W
                                                g : Quiver.Hom W X
                                                h : Quiver.Hom W Y
                                                ⊢ Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.ChosenFiniteProduct …
                                              -/
                                                      /-
                                                        🎉 no goals
                                                      -/
    f ≫ lift g h = lift (f ≫ g) (f ≫ h) := by ext <;> simp
                                                      /-
                                                        🎉 no goals
                                                      -/


@[simp]
                                                                          /-
                                                                            C : Type u
                                                                            inst✝¹ : CategoryTheory.Category.{v, u} C
                                                                            inst✝ : CategoryTheory.ChosenFiniteProducts C
                                                                            X Y : C
                                                                            ⊢ Eq (CategoryTheory.ChosenFiniteProducts.lift (CategoryTheory.ChosenFinitePro …
                                                                          -/
                                                                                  /-
                                                                                    🎉 no goals
                                                                                  -/
lemma lift_fst_snd {X Y : C} : lift (fst X Y) (snd X Y) = 𝟙 (X ⊗ Y) := by ext <;> simp
                                                                                  /-
                                                                                    🎉 no goals
                                                                                  -/


@[reassoc (attr := simp)]
lemma tensorHom_fst {X₁ X₂ Y₁ Y₂ : C} (f : X₁ ⟶ X₂) (g : Y₁ ⟶ Y₂) :
    (f ⊗ g) ≫ fst _ _ = fst _ _ ≫ f := lift_fst _ _


@[reassoc (attr := simp)]
lemma tensorHom_snd {X₁ X₂ Y₁ Y₂ : C} (f : X₁ ⟶ X₂) (g : Y₁ ⟶ Y₂) :
    (f ⊗ g) ≫ snd _ _ = snd _ _ ≫ g := lift_snd _ _


@[reassoc (attr := simp)]
lemma lift_map {V W X Y Z : C} (f : V ⟶ W) (g : V ⟶ X) (h : W ⟶ Y) (k : X ⟶ Z) :
                                                    /-
                                                      C : Type u
                                                      inst✝¹ : CategoryTheory.Category.{v, u} C
                                                      inst✝ : CategoryTheory.ChosenFiniteProducts C
                                                      V W X Y Z : C
                                                      f : Quiver.Hom V W
                                                      g : Quiver.Hom V X
                                                      h : Quiver.Hom W Y
                                                      k : Quiver.Hom X Z
                                                      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.ChosenFiniteProducts. …
                                                    -/
                                                            /-
                                                              🎉 no goals
                                                            -/
    lift f g ≫ (h ⊗ k) = lift (f ≫ h) (g ≫ k) := by ext <;> simp
                                                            /-
                                                              🎉 no goals
                                                            -/


@[simp]
lemma lift_fst_comp_snd_comp {W X Y Z : C} (g : W ⟶ X) (g' : Y ⟶ Z) :
                                                     /-
                                                       C : Type u
                                                       inst✝¹ : CategoryTheory.Category.{v, u} C
                                                       inst✝ : CategoryTheory.ChosenFiniteProducts C
                                                       W X Y Z : C
                                                       g : Quiver.Hom W X
                                                       g' : Quiver.Hom Y Z
                                                       ⊢ Eq (CategoryTheory.ChosenFiniteProducts.lift (CategoryTheory.CategoryStruct. …
                                                     -/
                                                             /-
                                                               🎉 no goals
                                                             -/
    lift (fst _ _ ≫ g) (snd _ _ ≫ g') = g ⊗ g' := by ext <;> simp
                                                             /-
                                                               🎉 no goals
                                                             -/


@[reassoc (attr := simp)]
lemma whiskerLeft_fst (X : C) {Y₁ Y₂ : C} (g : Y₁ ⟶ Y₂) :
    (X ◁ g) ≫ fst _ _ = fst _ _ :=
                                /-
                                  C : Type u
                                  inst✝¹ : CategoryTheory.Category.{v, u} C
                                  inst✝ : CategoryTheory.ChosenFiniteProducts C
                                  X Y₁ Y₂ : C
                                  g : Quiver.Hom Y₁ Y₂
                                  ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.ChosenFiniteProducts. …
                                -/
  (tensorHom_fst _ _).trans (by simp)
                                /-
                                  🎉 no goals
                                -/


@[reassoc (attr := simp)]
lemma whiskerLeft_snd (X : C) {Y₁ Y₂ : C} (g : Y₁ ⟶ Y₂) :
    (X ◁ g) ≫ snd _ _ = snd _ _ ≫ g :=
  tensorHom_snd _ _


@[reassoc (attr := simp)]
lemma whiskerRight_fst {X₁ X₂ : C} (f : X₁ ⟶ X₂) (Y : C) :
    (f ▷ Y) ≫ fst _ _ = fst _ _ ≫ f :=
  tensorHom_fst _ _


@[reassoc (attr := simp)]
lemma whiskerRight_snd {X₁ X₂ : C} (f : X₁ ⟶ X₂) (Y : C) :
    (f ▷ Y) ≫ snd _ _ = snd _ _ :=
                                /-
                                  C : Type u
                                  inst✝¹ : CategoryTheory.Category.{v, u} C
                                  inst✝ : CategoryTheory.ChosenFiniteProducts C
                                  X₁ X₂ : C
                                  f : Quiver.Hom X₁ X₂
                                  Y : C
                                  ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.ChosenFiniteProducts. …
                                -/
  (tensorHom_snd _ _).trans (by simp)
                                /-
                                  🎉 no goals
                                -/


@[reassoc (attr := simp)]
lemma associator_hom_fst (X Y Z : C) :
    (α_ X Y Z).hom ≫ fst _ _ = fst _ _ ≫ fst _ _ := lift_fst _ _


@[reassoc (attr := simp)]
lemma associator_hom_snd_fst (X Y Z : C) :
    (α_ X Y Z).hom ≫ snd _ _ ≫ fst _ _ = fst _ _ ≫ snd _ _  := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.ChosenFiniteProducts C
    X Y Z : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  erw [lift_snd_assoc, lift_fst]
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.ChosenFiniteProducts C
    X Y Z : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.BinaryFan.fst  …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma associator_hom_snd_snd (X Y Z : C) :
    (α_ X Y Z).hom ≫ snd _ _ ≫ snd _ _ = snd _ _  := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.ChosenFiniteProducts C
    X Y Z : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  erw [lift_snd_assoc, lift_snd]
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.ChosenFiniteProducts C
    X Y Z : C
    ⊢ Eq (CategoryTheory.Limits.BinaryFan.snd (CategoryTheory.ChosenFiniteProducts …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma associator_inv_fst (X Y Z : C) :
    (α_ X Y Z).inv ≫ fst _ _ ≫ fst _ _ = fst _ _ := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.ChosenFiniteProducts C
    X Y Z : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  erw [lift_fst_assoc, lift_fst]
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.ChosenFiniteProducts C
    X Y Z : C
    ⊢ Eq (CategoryTheory.Limits.BinaryFan.fst (CategoryTheory.ChosenFiniteProducts …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma associator_inv_fst_snd (X Y Z : C) :
    (α_ X Y Z).inv ≫ fst _ _ ≫ snd _ _ = snd _ _ ≫ fst _ _ := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.ChosenFiniteProducts C
    X Y Z : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  erw [lift_fst_assoc, lift_snd]
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.ChosenFiniteProducts C
    X Y Z : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.BinaryFan.snd  …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma associator_inv_snd (X Y Z : C) :
    (α_ X Y Z).inv ≫ snd _ _ = snd _ _ ≫ snd _ _ := lift_snd _ _


@[reassoc (attr := simp)]
lemma leftUnitor_inv_fst (X : C) :
    (λ_ X).inv ≫ fst _ _ = toUnit _ := toUnit_unique _ _


@[reassoc (attr := simp)]
lemma leftUnitor_inv_snd (X : C) :
    (λ_ X).inv ≫ snd _ _ = 𝟙 X := lift_snd _ _


@[reassoc (attr := simp)]
lemma rightUnitor_inv_fst (X : C) :
    (ρ_ X).inv ≫ fst _ _ = 𝟙 X := lift_fst _ _


@[reassoc (attr := simp)]
lemma rightUnitor_inv_snd (X : C) :
    (ρ_ X).inv ≫ snd _ _ = toUnit _ := toUnit_unique _ _


/--
Construct an instance of `ChosenFiniteProducts C` given an instance of `HasFiniteProducts C`.
-/
noncomputable
def ofFiniteProducts
    (C : Type u) [Category.{v} C] [Limits.HasFiniteProducts C] :
    ChosenFiniteProducts C where
  product X Y := Limits.getLimitCone (Limits.pair X Y)
  terminal := Limits.getLimitCone (Functor.empty C)


instance (priority := 100) : Limits.HasFiniteProducts C :=
  letI : ∀ (X Y : C), Limits.HasLimit (Limits.pair X Y) := fun _ _ =>
    .mk <| ChosenFiniteProducts.product _ _
  letI : Limits.HasBinaryProducts C := Limits.hasBinaryProducts_of_hasLimit_pair _
  letI : Limits.HasTerminal C := Limits.hasTerminal_of_unique (𝟙_ C)
  hasFiniteProducts_of_has_binary_and_terminal


/-- When `C` and `D` have chosen finite products and `F : C ⥤ D` is any functor,
`terminalComparison F` is the unique map `F (𝟙_ C) ⟶ 𝟙_ D`. -/
abbrev terminalComparison : F.obj (𝟙_ C) ⟶ 𝟙_ D := toUnit _


@[reassoc (attr := simp)]
lemma map_toUnit_comp_terminalCompariso (A : C) :
    F.map (toUnit A) ≫ terminalComparison F = toUnit _ := toUnit_unique _ _


/-- If `terminalComparison F` is an Iso, then `F` preserves terminal objects. -/
lemma preservesLimit_empty_of_isIso_terminalComparison [IsIso (terminalComparison F)] :
    PreservesLimit (Functor.empty.{0} C) F := by
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.ChosenFiniteProducts C
    D : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} D
    inst✝¹ : CategoryTheory.ChosenFiniteProducts D
    F : CategoryTheory.Functor C D
    inst✝ : CategoryTheory.IsIso (CategoryTheory.ChosenFiniteProducts.terminalComp …
    ⊢ CategoryTheory.Limits.PreservesLimit (CategoryTheory.Functor.empty C) F
  -/
  apply preservesLimit_of_preserves_limit_cone terminal.isLimit
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.ChosenFiniteProducts C
    D : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} D
    inst✝¹ : CategoryTheory.ChosenFiniteProducts D
    F : CategoryTheory.Functor C D
    inst✝ : CategoryTheory.IsIso (CategoryTheory.ChosenFiniteProducts.terminalComp …
    ⊢ CategoryTheory.Limits.IsLimit (F.mapCone CategoryTheory.ChosenFiniteProducts …
  -/
  apply isLimitChangeEmptyCone D terminal.isLimit
  /-
    case hi
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.ChosenFiniteProducts C
    D : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} D
    inst✝¹ : CategoryTheory.ChosenFiniteProducts D
    F : CategoryTheory.Functor C D
    inst✝ : CategoryTheory.IsIso (CategoryTheory.ChosenFiniteProducts.terminalComp …
    ⊢ CategoryTheory.Iso CategoryTheory.ChosenFiniteProducts.terminal.cone.pt (F.m …
  -/
  exact asIso (terminalComparison F)|>.symm
  /-
    🎉 no goals
  -/


/-- If `F` preserves terminal objects, then `terminalComparison F` is an isomorphism. -/
noncomputable def preservesTerminalIso [h : PreservesLimit (Functor.empty.{0} C) F] :
    F.obj (𝟙_ C) ≅ 𝟙_ D :=
  (isLimitChangeEmptyCone D (isLimitOfPreserves _ terminal.isLimit) (asEmptyCone (F.obj (𝟙_ C)))
    (Iso.refl _)).conePointUniqueUpToIso terminal.isLimit


@[simp]
lemma preservesTerminalIso_hom [PreservesLimit (Functor.empty.{0} C) F] :
    (preservesTerminalIso F).hom = terminalComparison F := toUnit_unique _ _


instance terminalComparison_isIso_of_preservesLimits [PreservesLimit (Functor.empty.{0} C) F] :
    IsIso (terminalComparison F) := by
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.ChosenFiniteProducts C
    D : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} D
    inst✝¹ : CategoryTheory.ChosenFiniteProducts D
    F : CategoryTheory.Functor C D
    inst✝ : CategoryTheory.Limits.PreservesLimit (CategoryTheory.Functor.empty C) F
    ⊢ CategoryTheory.IsIso (CategoryTheory.ChosenFiniteProducts.terminalComparison …
  -/
  rw [← preservesTerminalIso_hom]
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.ChosenFiniteProducts C
    D : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} D
    inst✝¹ : CategoryTheory.ChosenFiniteProducts D
    F : CategoryTheory.Functor C D
    inst✝ : CategoryTheory.Limits.PreservesLimit (CategoryTheory.Functor.empty C) F
    ⊢ CategoryTheory.IsIso (CategoryTheory.ChosenFiniteProducts.preservesTerminalI …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


/-- When `C` and `D` have chosen finite products and `F : C ⥤ D` is any functor,
`prodComparison F A B` is the canonical comparison morphism from `F (A ⊗ B)` to `F(A) ⊗ F(B)`. -/
def prodComparison (A B : C) : F.obj (A ⊗ B) ⟶ F.obj A ⊗ F.obj B :=
  lift (F.map (fst A B)) (F.map (snd A B))


@[reassoc (attr := simp)]
theorem prodComparison_fst : prodComparison F A B ≫ fst _ _ = F.map (fst A B) :=
  lift_fst _ _


@[reassoc (attr := simp)]
theorem prodComparison_snd : prodComparison F A B ≫ snd _ _ = F.map (snd A B) :=
  lift_snd _ _


@[reassoc (attr := simp)]
theorem inv_prodComparison_map_fst [IsIso (prodComparison F A B)] :
                                                                 /-
                                                                   C : Type u
                                                                   inst✝⁴ : CategoryTheory.Category.{v, u} C
                                                                   inst✝³ : CategoryTheory.ChosenFiniteProducts C
                                                                   D : Type u₁
                                                                   inst✝² : CategoryTheory.Category.{v₁, u₁} D
                                                                   inst✝¹ : CategoryTheory.ChosenFiniteProducts D
                                                                   F : CategoryTheory.Functor C D
                                                                   A B : C
                                                                   inst✝ : CategoryTheory.IsIso (CategoryTheory.ChosenFiniteProducts.prodComparis …
                                                                   ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.inv (CategoryTheory.C …
                                                                 -/
    inv (prodComparison F A B) ≫ F.map (fst _ _) = fst _ _ := by simp [IsIso.inv_comp_eq]
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


@[reassoc (attr := simp)]
theorem inv_prodComparison_map_snd [IsIso (prodComparison F A B)] :
                                                                 /-
                                                                   C : Type u
                                                                   inst✝⁴ : CategoryTheory.Category.{v, u} C
                                                                   inst✝³ : CategoryTheory.ChosenFiniteProducts C
                                                                   D : Type u₁
                                                                   inst✝² : CategoryTheory.Category.{v₁, u₁} D
                                                                   inst✝¹ : CategoryTheory.ChosenFiniteProducts D
                                                                   F : CategoryTheory.Functor C D
                                                                   A B : C
                                                                   inst✝ : CategoryTheory.IsIso (CategoryTheory.ChosenFiniteProducts.prodComparis …
                                                                   ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.inv (CategoryTheory.C …
                                                                 -/
    inv (prodComparison F A B) ≫ F.map (snd _ _) = snd _ _ := by simp [IsIso.inv_comp_eq]
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


/-- Naturality of the `prodComparison` morphism in both arguments. -/
@[reassoc]
theorem prodComparison_natural (f : A ⟶ A') (g : B ⟶ B') :
    F.map (f ⊗ g) ≫ prodComparison F A' B' =
      prodComparison F A B ≫ (F.map f ⊗ F.map g) := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.ChosenFiniteProducts C
    D : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} D
    inst✝ : CategoryTheory.ChosenFiniteProducts D
    F : CategoryTheory.Functor C D
    A B A' B' : C
    f : Quiver.Hom A A'
    g : Quiver.Hom B B'
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map (CategoryTheory.MonoidalCatego …
  -/
  apply hom_ext <;>
  simp only [Category.assoc, prodComparison_fst, tensorHom_fst, prodComparison_fst_assoc,
    prodComparison_snd, tensorHom_snd, prodComparison_snd_assoc, ← F.map_comp]


/-- Naturality of the `prodComparison` morphism in the right argument. -/
@[reassoc]
theorem prodComparison_natural_whiskerLeft (g : B ⟶ B') :
    F.map (A ◁ g) ≫ prodComparison F A B' =
      prodComparison F A B ≫ (F.obj A ◁ F.map g) := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.ChosenFiniteProducts C
    D : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} D
    inst✝ : CategoryTheory.ChosenFiniteProducts D
    F : CategoryTheory.Functor C D
    A B B' : C
    g : Quiver.Hom B B'
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map (CategoryTheory.MonoidalCatego …
  -/
  rw [← id_tensorHom, prodComparison_natural, Functor.map_id]
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.ChosenFiniteProducts C
    D : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} D
    inst✝ : CategoryTheory.ChosenFiniteProducts D
    F : CategoryTheory.Functor C D
    A B B' : C
    g : Quiver.Hom B B'
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.ChosenFiniteProducts. …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- Naturality of the `prodComparison` morphism in the left argument. -/
@[reassoc]
theorem prodComparison_natural_whiskerRight (f : A ⟶ A') :
    F.map (f ▷ B) ≫ prodComparison F A' B =
      prodComparison F A B ≫ (F.map f ▷ F.obj B) := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.ChosenFiniteProducts C
    D : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} D
    inst✝ : CategoryTheory.ChosenFiniteProducts D
    F : CategoryTheory.Functor C D
    A B A' : C
    f : Quiver.Hom A A'
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map (CategoryTheory.MonoidalCatego …
  -/
  rw [← tensorHom_id, prodComparison_natural, Functor.map_id]
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.ChosenFiniteProducts C
    D : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} D
    inst✝ : CategoryTheory.ChosenFiniteProducts D
    F : CategoryTheory.Functor C D
    A B A' : C
    f : Quiver.Hom A A'
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.ChosenFiniteProducts. …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- If the product comparison morphism is an iso, its inverse is natural in both argument. -/
@[reassoc]
theorem prodComparison_inv_natural (f : A ⟶ A') (g : B ⟶ B') [IsIso (prodComparison F A' B')] :
    inv (prodComparison F A B) ≫ F.map (f ⊗ g) =
      (F.map f ⊗ F.map g) ≫ inv (prodComparison F A' B') := by
  /-
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    inst✝⁴ : CategoryTheory.ChosenFiniteProducts C
    D : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} D
    inst✝² : CategoryTheory.ChosenFiniteProducts D
    F : CategoryTheory.Functor C D
    A B A' B' : C
    inst✝¹ : CategoryTheory.IsIso (CategoryTheory.ChosenFiniteProducts.prodCompari …
    f : Quiver.Hom A A'
    g : Quiver.Hom B B'
    inst✝ : CategoryTheory.IsIso (CategoryTheory.ChosenFiniteProducts.prodComparis …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.inv (CategoryTheory.C …
  -/
  rw [IsIso.eq_comp_inv, Category.assoc, IsIso.inv_comp_eq, prodComparison_natural]
  /-
    🎉 no goals
  -/


/-- If the product comparison morphism is an iso, its inverse is natural in the right argument. -/
@[reassoc]
theorem prodComparison_inv_natural_whiskerLeft (g : B ⟶ B') [IsIso (prodComparison F A B')] :
    inv (prodComparison F A B) ≫ F.map (A ◁ g) =
      (F.obj A ◁ F.map g) ≫ inv (prodComparison F A B') := by
  /-
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    inst✝⁴ : CategoryTheory.ChosenFiniteProducts C
    D : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} D
    inst✝² : CategoryTheory.ChosenFiniteProducts D
    F : CategoryTheory.Functor C D
    A B B' : C
    inst✝¹ : CategoryTheory.IsIso (CategoryTheory.ChosenFiniteProducts.prodCompari …
    g : Quiver.Hom B B'
    inst✝ : CategoryTheory.IsIso (CategoryTheory.ChosenFiniteProducts.prodComparis …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.inv (CategoryTheory.C …
  -/
  rw [IsIso.eq_comp_inv, Category.assoc, IsIso.inv_comp_eq, prodComparison_natural_whiskerLeft]
  /-
    🎉 no goals
  -/


/-- If the product comparison morphism is an iso, its inverse is natural in the left argument. -/
@[reassoc]
theorem prodComparison_inv_natural_whiskerRight (f : A ⟶ A') [IsIso (prodComparison F A' B)] :
    inv (prodComparison F A B) ≫ F.map (f ▷ B) =
      (F.map f ▷ F.obj B) ≫ inv (prodComparison F A' B) := by
  /-
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    inst✝⁴ : CategoryTheory.ChosenFiniteProducts C
    D : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} D
    inst✝² : CategoryTheory.ChosenFiniteProducts D
    F : CategoryTheory.Functor C D
    A B A' : C
    inst✝¹ : CategoryTheory.IsIso (CategoryTheory.ChosenFiniteProducts.prodCompari …
    f : Quiver.Hom A A'
    inst✝ : CategoryTheory.IsIso (CategoryTheory.ChosenFiniteProducts.prodComparis …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.inv (CategoryTheory.C …
  -/
  rw [IsIso.eq_comp_inv, Category.assoc, IsIso.inv_comp_eq, prodComparison_natural_whiskerRight]
  /-
    🎉 no goals
  -/


theorem prodComparison_comp {E : Type u₂} [Category.{v₂} E] [ChosenFiniteProducts E] (G : D ⥤ E) :
    prodComparison (F ⋙ G) A B =
      G.map (prodComparison F A B) ≫ prodComparison G (F.obj A) (F.obj B) := by
  /-
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    inst✝⁴ : CategoryTheory.ChosenFiniteProducts C
    D : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} D
    inst✝² : CategoryTheory.ChosenFiniteProducts D
    F : CategoryTheory.Functor C D
    A B : C
    E : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} E
    inst✝ : CategoryTheory.ChosenFiniteProducts E
    G : CategoryTheory.Functor D E
    ⊢ Eq (CategoryTheory.ChosenFiniteProducts.prodComparison (F.comp G) A B) (Cate …
  -/
  unfold prodComparison
  /-
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    inst✝⁴ : CategoryTheory.ChosenFiniteProducts C
    D : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} D
    inst✝² : CategoryTheory.ChosenFiniteProducts D
    F : CategoryTheory.Functor C D
    A B : C
    E : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} E
    inst✝ : CategoryTheory.ChosenFiniteProducts E
    G : CategoryTheory.Functor D E
    ⊢ Eq (CategoryTheory.ChosenFiniteProducts.lift ((F.comp G).map (CategoryTheory …
  -/
          /-
            🎉 no goals
          -/
  ext <;> simp [← G.map_comp]
          /-
            🎉 no goals
          -/


@[simp]
lemma prodComparison_id :
    prodComparison (𝟭 C) A B = 𝟙 (A ⊗ B) := lift_fst_snd


/-- The product comparison morphism from `F(A ⊗ -)` to `FA ⊗ F-`, whose components are given by
`prodComparison`. -/
@[simps]
def prodComparisonNatTrans (A : C) :
    (curriedTensor C).obj A ⋙ F ⟶ F ⋙ (curriedTensor D).obj (F.obj A) where
  app B := prodComparison F A B
  naturality x y f := by
    /-
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.ChosenFiniteProducts C
      D : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} D
      inst✝ : CategoryTheory.ChosenFiniteProducts D
      F : CategoryTheory.Functor C D
      A✝ B A' B' A x y : C
      f : Quiver.Hom x y
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((((CategoryTheory.MonoidalCategory.c …
    -/
    apply hom_ext <;>
    simp only [Functor.comp_obj, curriedTensor_obj_obj,
      Functor.comp_map, curriedTensor_obj_map, Category.assoc, prodComparison_fst, whiskerLeft_fst,
      prodComparison_snd, prodComparison_snd_assoc, whiskerLeft_snd, ← F.map_comp]


theorem prodComparisonNatTrans_comp {E : Type u₂} [Category.{v₂} E] [ChosenFiniteProducts E]
    (G : D ⥤ E) : prodComparisonNatTrans (F ⋙ G) A =
      whiskerRight (prodComparisonNatTrans F A) G ≫
                                                                 /-
                                                                   C : Type u
                                                                   inst✝⁵ : CategoryTheory.Category.{v, u} C
                                                                   inst✝⁴ : CategoryTheory.ChosenFiniteProducts C
                                                                   D : Type u₁
                                                                   inst✝³ : CategoryTheory.Category.{v₁, u₁} D
                                                                   inst✝² : CategoryTheory.ChosenFiniteProducts D
                                                                   F : CategoryTheory.Functor C D
                                                                   A : C
                                                                   E : Type u₂
                                                                   inst✝¹ : CategoryTheory.Category.{v₂, u₂} E
                                                                   inst✝ : CategoryTheory.ChosenFiniteProducts E
                                                                   G : CategoryTheory.Functor D E
                                                                   ⊢ Eq (CategoryTheory.ChosenFiniteProducts.prodComparisonNatTrans (F.comp G) A) …
                                                                 -/
        whiskerLeft F (prodComparisonNatTrans G (F.obj A)) := by ext; simp [prodComparison_comp]
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


@[simp]
lemma prodComparisonNatTrans_id :
                                               /-
                                                 C : Type u
                                                 inst✝¹ : CategoryTheory.Category.{v, u} C
                                                 inst✝ : CategoryTheory.ChosenFiniteProducts C
                                                 A : C
                                                 ⊢ Eq (CategoryTheory.ChosenFiniteProducts.prodComparisonNatTrans (CategoryTheo …
                                               -/
    prodComparisonNatTrans (𝟭 C) A = 𝟙 _ := by ext; simp
                                                    /-
                                                      🎉 no goals
                                                    -/


/-- The product comparison morphism from `F(- ⊗ -)` to `F- ⊗ F-`, whose components are given by
`prodComparison`. -/
@[simps]
def prodComparisonBifunctorNatTrans :
    curriedTensor C ⋙ (whiskeringRight _ _ _).obj F ⟶
      F ⋙ curriedTensor D ⋙ (whiskeringLeft _ _ _).obj F where
  app A := prodComparisonNatTrans F A
  naturality x y f := by
    /-
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.ChosenFiniteProducts C
      D : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} D
      inst✝ : CategoryTheory.ChosenFiniteProducts D
      F : CategoryTheory.Functor C D
      A B A' B' x y : C
      f : Quiver.Hom x y
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.MonoidalCategory.cu …
    -/
    ext z
    /-
      case w.h
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.ChosenFiniteProducts C
      D : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} D
      inst✝ : CategoryTheory.ChosenFiniteProducts D
      F : CategoryTheory.Functor C D
      A B A' B' x y : C
      f : Quiver.Hom x y
      z : C
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp (((CategoryTheory.MonoidalCategory.c …
    -/
                      /-
                        🎉 no goals
                      -/
    apply hom_ext <;> simp [← Functor.map_comp]
                      /-
                        🎉 no goals
                      -/


theorem prodComparisonBifunctorNatTrans_comp {E : Type u₂} [Category.{v₂} E]
    [ChosenFiniteProducts E] (G : D ⥤ E) : prodComparisonBifunctorNatTrans (F ⋙ G) =
      whiskerRight (prodComparisonBifunctorNatTrans F) ((whiskeringRight _ _ _).obj G) ≫
        whiskerLeft F (whiskerRight (prodComparisonBifunctorNatTrans G)
                                                /-
                                                  C : Type u
                                                  inst✝⁵ : CategoryTheory.Category.{v, u} C
                                                  inst✝⁴ : CategoryTheory.ChosenFiniteProducts C
                                                  D : Type u₁
                                                  inst✝³ : CategoryTheory.Category.{v₁, u₁} D
                                                  inst✝² : CategoryTheory.ChosenFiniteProducts D
                                                  F : CategoryTheory.Functor C D
                                                  E : Type u₂
                                                  inst✝¹ : CategoryTheory.Category.{v₂, u₂} E
                                                  inst✝ : CategoryTheory.ChosenFiniteProducts E
                                                  G : CategoryTheory.Functor D E
                                                  ⊢ Eq (CategoryTheory.ChosenFiniteProducts.prodComparisonBifunctorNatTrans (F.c …
                                                -/
          ((whiskeringLeft _ _ _).obj F)) := by ext; simp [prodComparison_comp]
                                                     /-
                                                       🎉 no goals
                                                     -/


instance (A : C) [∀ B, IsIso (prodComparison F A B)] : IsIso (prodComparisonNatTrans F A) := by
  /-
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    inst✝⁵ : CategoryTheory.ChosenFiniteProducts C
    D : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} D
    inst✝³ : CategoryTheory.ChosenFiniteProducts D
    F : CategoryTheory.Functor C D
    A✝ B A' B' : C
    E : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} E
    inst✝¹ : CategoryTheory.ChosenFiniteProducts E
    G : CategoryTheory.Functor D E
    A : C
    inst✝ : ∀ (B : C), CategoryTheory.IsIso (CategoryTheory.ChosenFiniteProducts.p …
    ⊢ CategoryTheory.IsIso (CategoryTheory.ChosenFiniteProducts.prodComparisonNatT …
  -/
  letI : ∀ X, IsIso ((prodComparisonNatTrans F A).app X) := by assumption
  /-
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    inst✝⁵ : CategoryTheory.ChosenFiniteProducts C
    D : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} D
    inst✝³ : CategoryTheory.ChosenFiniteProducts D
    F : CategoryTheory.Functor C D
    A✝ B A' B' : C
    E : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} E
    inst✝¹ : CategoryTheory.ChosenFiniteProducts E
    G : CategoryTheory.Functor D E
    A : C
    inst✝ : ∀ (B : C), CategoryTheory.IsIso (CategoryTheory.ChosenFiniteProducts.p …
    this : ∀ (X : C), CategoryTheory.IsIso ((CategoryTheory.ChosenFiniteProducts.p …
    ⊢ CategoryTheory.IsIso (CategoryTheory.ChosenFiniteProducts.prodComparisonNatT …
  -/
  apply NatIso.isIso_of_isIso_app
  /-
    🎉 no goals
  -/


instance [∀ A B, IsIso (prodComparison F A B)] : IsIso (prodComparisonBifunctorNatTrans F) := by
  letI : ∀ X, IsIso ((prodComparisonBifunctorNatTrans F).app X) :=
    fun _ ↦ by dsimp; apply NatIso.isIso_of_isIso_app
  /-
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    inst✝⁵ : CategoryTheory.ChosenFiniteProducts C
    D : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} D
    inst✝³ : CategoryTheory.ChosenFiniteProducts D
    F : CategoryTheory.Functor C D
    A B A' B' : C
    E : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} E
    inst✝¹ : CategoryTheory.ChosenFiniteProducts E
    G : CategoryTheory.Functor D E
    inst✝ : ∀ (A B : C), CategoryTheory.IsIso (CategoryTheory.ChosenFiniteProducts …
    this : ∀ (X : C), CategoryTheory.IsIso ((CategoryTheory.ChosenFiniteProducts.p …
    ⊢ CategoryTheory.IsIso (CategoryTheory.ChosenFiniteProducts.prodComparisonBifu …
  -/
  apply NatIso.isIso_of_isIso_app
  /-
    🎉 no goals
  -/


/-- If `F` preserves the limit of the pair `(A, B)`, then the binary fan given by
`(F.map fst A B, F.map (snd A B))` is a limit cone. -/
noncomputable def isLimitChosenFiniteProductsOfPreservesLimits :
    IsLimit <| BinaryFan.mk (F.map (fst A B)) (F.map (snd A B)) :=
  mapIsLimitOfPreservesOfIsLimit F (fst _ _) (snd _ _) <|
    (product A B).isLimit.ofIsoLimit <| isoBinaryFanMk (product A B).cone


/-- If `F` preserves the limit of the pair `(A, B)`, then `prodComparison F A B` is an isomorphism.
-/
noncomputable def prodComparisonIso : F.obj (A ⊗ B) ≅ F.obj A ⊗ F.obj B :=
  IsLimit.conePointUniqueUpToIso (isLimitChosenFiniteProductsOfPreservesLimits F A B)
    (product _ _).isLimit


@[simp]
lemma prodComparisonIso_hom : (prodComparisonIso F A B).hom = prodComparison F A B := by
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.ChosenFiniteProducts C
    D : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} D
    inst✝¹ : CategoryTheory.ChosenFiniteProducts D
    F : CategoryTheory.Functor C D
    A B : C
    inst✝ : CategoryTheory.Limits.PreservesLimit (CategoryTheory.Limits.pair A B) F
    ⊢ Eq (CategoryTheory.ChosenFiniteProducts.prodComparisonIso F A B).hom (Catego …
  -/
  rfl
  /-
    🎉 no goals
  -/


instance isIso_prodComparison_of_preservesLimit_pair : IsIso (prodComparison F A B) := by
  /-
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    inst✝⁵ : CategoryTheory.ChosenFiniteProducts C
    D : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} D
    inst✝³ : CategoryTheory.ChosenFiniteProducts D
    F : CategoryTheory.Functor C D
    A B A' B' : C
    E : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} E
    inst✝¹ : CategoryTheory.ChosenFiniteProducts E
    G : CategoryTheory.Functor D E
    inst✝ : CategoryTheory.Limits.PreservesLimit (CategoryTheory.Limits.pair A B) F
    ⊢ CategoryTheory.IsIso (CategoryTheory.ChosenFiniteProducts.prodComparison F A …
  -/
  rw [← prodComparisonIso_hom]
  /-
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    inst✝⁵ : CategoryTheory.ChosenFiniteProducts C
    D : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} D
    inst✝³ : CategoryTheory.ChosenFiniteProducts D
    F : CategoryTheory.Functor C D
    A B A' B' : C
    E : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} E
    inst✝¹ : CategoryTheory.ChosenFiniteProducts E
    G : CategoryTheory.Functor D E
    inst✝ : CategoryTheory.Limits.PreservesLimit (CategoryTheory.Limits.pair A B) F
    ⊢ CategoryTheory.IsIso (CategoryTheory.ChosenFiniteProducts.prodComparisonIso  …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


/-- The natural isomorphism `F(A ⊗ -) ≅ FA ⊗ F-`, provided each `prodComparison F A B` is an
isomorphism (as `B` changes). -/
@[simps! hom inv]
noncomputable def prodComparisonNatIso (A : C) [∀ B, PreservesLimit (pair A B) F] :
    (curriedTensor C).obj A ⋙ F ≅ F ⋙ (curriedTensor D).obj (F.obj A) :=
  asIso (prodComparisonNatTrans F A)


/-- The natural isomorphism of bifunctors `F(- ⊗ -) ≅ F- ⊗ F-`, provided each
`prodComparison F A B` is an isomorphism. -/
@[simps! hom inv]
noncomputable def prodComparisonBifunctorNatIso [∀ A B, PreservesLimit (pair A B) F] :
    curriedTensor C ⋙ (whiskeringRight _ _ _).obj F ≅
      F ⋙ curriedTensor D ⋙ (whiskeringLeft _ _ _).obj F :=
  asIso (prodComparisonBifunctorNatTrans F)


/-- If `prodComparison F A B` is an isomorphism, then `F` preserves the limit of `pair A B`. -/
lemma preservesLimit_pair_of_isIso_prodComparison (A B : C)
    [IsIso (prodComparison F A B)] :
    PreservesLimit (pair A B) F := by
 /-
   C : Type u
   inst✝⁴ : CategoryTheory.Category.{v, u} C
   inst✝³ : CategoryTheory.ChosenFiniteProducts C
   D : Type u₁
   inst✝² : CategoryTheory.Category.{v₁, u₁} D
   inst✝¹ : CategoryTheory.ChosenFiniteProducts D
   F : CategoryTheory.Functor C D
   A B : C
   inst✝ : CategoryTheory.IsIso (CategoryTheory.ChosenFiniteProducts.prodComparis …
   ⊢ CategoryTheory.Limits.PreservesLimit (CategoryTheory.Limits.pair A B) F
 -/
 apply preservesLimit_of_preserves_limit_cone (product A B).isLimit
 refine IsLimit.equivOfNatIsoOfIso (pairComp A B F) _
    ((product (F.obj A) (F.obj B)).cone.extend (prodComparison F A B))
      (BinaryFan.ext (by exact Iso.refl _) ?_ ?_) |>.invFun
      (IsLimit.extendIso _ (product (F.obj A) (F.obj B)).isLimit)
   /-
     case refine_1
     C : Type u
     inst✝⁴ : CategoryTheory.Category.{v, u} C
     inst✝³ : CategoryTheory.ChosenFiniteProducts C
     D : Type u₁
     inst✝² : CategoryTheory.Category.{v₁, u₁} D
     inst✝¹ : CategoryTheory.ChosenFiniteProducts D
     F : CategoryTheory.Functor C D
     A B : C
     inst✝ : CategoryTheory.IsIso (CategoryTheory.ChosenFiniteProducts.prodComparis …
     ⊢ Eq (CategoryTheory.Limits.BinaryFan.fst ((CategoryTheory.Limits.Cones.postco …
   -/
 · dsimp only [BinaryFan.fst]
   /-
     case refine_1
     C : Type u
     inst✝⁴ : CategoryTheory.Category.{v, u} C
     inst✝³ : CategoryTheory.ChosenFiniteProducts C
     D : Type u₁
     inst✝² : CategoryTheory.Category.{v₁, u₁} D
     inst✝¹ : CategoryTheory.ChosenFiniteProducts D
     F : CategoryTheory.Functor C D
     A B : C
     inst✝ : CategoryTheory.IsIso (CategoryTheory.ChosenFiniteProducts.prodComparis …
     ⊢ Eq (((CategoryTheory.Limits.Cones.postcompose (CategoryTheory.Limits.pairCom …
   -/
   simp [pairComp, prodComparison, lift, fst]
   /-
     🎉 no goals
   -/
   /-
     case refine_2
     C : Type u
     inst✝⁴ : CategoryTheory.Category.{v, u} C
     inst✝³ : CategoryTheory.ChosenFiniteProducts C
     D : Type u₁
     inst✝² : CategoryTheory.Category.{v₁, u₁} D
     inst✝¹ : CategoryTheory.ChosenFiniteProducts D
     F : CategoryTheory.Functor C D
     A B : C
     inst✝ : CategoryTheory.IsIso (CategoryTheory.ChosenFiniteProducts.prodComparis …
     ⊢ Eq (CategoryTheory.Limits.BinaryFan.snd ((CategoryTheory.Limits.Cones.postco …
   -/
 · dsimp only [BinaryFan.snd]
   /-
     case refine_2
     C : Type u
     inst✝⁴ : CategoryTheory.Category.{v, u} C
     inst✝³ : CategoryTheory.ChosenFiniteProducts C
     D : Type u₁
     inst✝² : CategoryTheory.Category.{v₁, u₁} D
     inst✝¹ : CategoryTheory.ChosenFiniteProducts D
     F : CategoryTheory.Functor C D
     A B : C
     inst✝ : CategoryTheory.IsIso (CategoryTheory.ChosenFiniteProducts.prodComparis …
     ⊢ Eq (((CategoryTheory.Limits.Cones.postcompose (CategoryTheory.Limits.pairCom …
   -/
   simp [pairComp, prodComparison, lift, snd]
   /-
     🎉 no goals
   -/

  
/-- If `prodComparison F A B` is an isomorphism for all `A B` then `F` preserves limits of shape
`Discrete (WalkingPair)`. -/
lemma preservesLimitsOfShape_discrete_walkingPair_of_isIso_prodComparison
    [∀ A B, IsIso (prodComparison F A B)] : PreservesLimitsOfShape (Discrete WalkingPair) F := by
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.ChosenFiniteProducts C
    D : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} D
    inst✝¹ : CategoryTheory.ChosenFiniteProducts D
    F : CategoryTheory.Functor C D
    inst✝ : ∀ (A B : C), CategoryTheory.IsIso (CategoryTheory.ChosenFiniteProducts …
    ⊢ CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete Catego …
  -/
  constructor
  /-
    case preservesLimit
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.ChosenFiniteProducts C
    D : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} D
    inst✝¹ : CategoryTheory.ChosenFiniteProducts D
    F : CategoryTheory.Functor C D
    inst✝ : ∀ (A B : C), CategoryTheory.IsIso (CategoryTheory.ChosenFiniteProducts …
    ⊢ autoParam (∀ {K : CategoryTheory.Functor (CategoryTheory.Discrete CategoryTh …
  -/
  intro K
  /-
    case preservesLimit
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.ChosenFiniteProducts C
    D : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} D
    inst✝¹ : CategoryTheory.ChosenFiniteProducts D
    F : CategoryTheory.Functor C D
    inst✝ : ∀ (A B : C), CategoryTheory.IsIso (CategoryTheory.ChosenFiniteProducts …
    K : CategoryTheory.Functor (CategoryTheory.Discrete CategoryTheory.Limits.Walk …
    ⊢ CategoryTheory.Limits.PreservesLimit K F
  -/
  refine @preservesLimit_of_iso_diagram _ _ _ _ _ _ _ _ _ (diagramIsoPair K).symm ?_
  /-
    case preservesLimit
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.ChosenFiniteProducts C
    D : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} D
    inst✝¹ : CategoryTheory.ChosenFiniteProducts D
    F : CategoryTheory.Functor C D
    inst✝ : ∀ (A B : C), CategoryTheory.IsIso (CategoryTheory.ChosenFiniteProducts …
    K : CategoryTheory.Functor (CategoryTheory.Discrete CategoryTheory.Limits.Walk …
    ⊢ CategoryTheory.Limits.PreservesLimit (CategoryTheory.Limits.pair (K.obj { as …
  -/
  apply preservesLimit_pair_of_isIso_prodComparison
  /-
    🎉 no goals
  -/


/-- Any functor between categories with chosen finite products induces an oplax monoial functor. -/
def oplaxMonoidalOfChosenFiniteProducts : F.OplaxMonoidal where
  η' := terminalComparison F
  δ' X Y := prodComparison F X Y
                             /-
                               C : Type u
                               inst✝³ : CategoryTheory.Category.{v, u} C
                               inst✝² : CategoryTheory.ChosenFiniteProducts C
                               D : Type u₁
                               inst✝¹ : CategoryTheory.Category.{v₁, u₁} D
                               inst✝ : CategoryTheory.ChosenFiniteProducts D
                               F : CategoryTheory.Functor C D
                               X✝ Y✝ : C
                               f : Quiver.Hom X✝ Y✝
                               X' : C
                               ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun X Y => CategoryTheory.ChosenFin …
                             -/
  δ'_natural_left f X' := by simpa using (prodComparison_natural F f (𝟙 X')).symm
                             /-
                               🎉 no goals
                             -/
                             /-
                               C : Type u
                               inst✝³ : CategoryTheory.Category.{v, u} C
                               inst✝² : CategoryTheory.ChosenFiniteProducts C
                               D : Type u₁
                               inst✝¹ : CategoryTheory.Category.{v₁, u₁} D
                               inst✝ : CategoryTheory.ChosenFiniteProducts D
                               F : CategoryTheory.Functor C D
                               X✝ Y✝ X : C
                               g : Quiver.Hom X✝ Y✝
                               ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun X Y => CategoryTheory.ChosenFin …
                             -/
  δ'_natural_right X g := by simpa using (prodComparison_natural F (𝟙 X) g).symm
                             /-
                               🎉 no goals
                             -/
  oplax_associativity' _ _ _ := by
    /-
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.ChosenFiniteProducts C
      D : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} D
      inst✝ : CategoryTheory.ChosenFiniteProducts D
      F : CategoryTheory.Functor C D
      x✝² x✝¹ x✝ : C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun X Y => CategoryTheory.ChosenFin …
    -/
    apply hom_ext
    /-
      case h_fst
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.ChosenFiniteProducts C
      D : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} D
      inst✝ : CategoryTheory.ChosenFiniteProducts D
      F : CategoryTheory.Functor C D
      x✝² x✝¹ x✝ : C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
    case' h_snd => apply hom_ext
    /-
      case h_snd.h_fst
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.ChosenFiniteProducts C
      D : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} D
      inst✝ : CategoryTheory.ChosenFiniteProducts D
      F : CategoryTheory.Functor C D
      x✝² x✝¹ x✝ : C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
    all_goals simp [← Functor.map_comp]
    /-
      🎉 no goals
    -/
  oplax_left_unitality' _ := by
    /-
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.ChosenFiniteProducts C
      D : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} D
      inst✝ : CategoryTheory.ChosenFiniteProducts D
      F : CategoryTheory.Functor C D
      x✝ : C
      ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.leftUnitor (F.obj x✝)).inv (Catego …
    -/
    apply hom_ext
      /-
        case h_fst
        C : Type u
        inst✝³ : CategoryTheory.Category.{v, u} C
        inst✝² : CategoryTheory.ChosenFiniteProducts C
        D : Type u₁
        inst✝¹ : CategoryTheory.Category.{v₁, u₁} D
        inst✝ : CategoryTheory.ChosenFiniteProducts D
        F : CategoryTheory.Functor C D
        x✝ : C
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
      -/
    · exact toUnit_unique _ _
      /-
        🎉 no goals
      -/
    · simp only [leftUnitor_inv_snd, Category.assoc, whiskerRight_snd,
        prodComparison_snd, ← F.map_comp, F.map_id]
  oplax_right_unitality' _ := by
    /-
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.ChosenFiniteProducts C
      D : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} D
      inst✝ : CategoryTheory.ChosenFiniteProducts D
      F : CategoryTheory.Functor C D
      x✝ : C
      ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.rightUnitor (F.obj x✝)).inv (Categ …
    -/
    apply hom_ext
    · simp only [rightUnitor_inv_fst, Category.assoc, whiskerLeft_fst,
        prodComparison_fst, ← F.map_comp, F.map_id]
      /-
        case h_snd
        C : Type u
        inst✝³ : CategoryTheory.Category.{v, u} C
        inst✝² : CategoryTheory.ChosenFiniteProducts C
        D : Type u₁
        inst✝¹ : CategoryTheory.Category.{v₁, u₁} D
        inst✝ : CategoryTheory.ChosenFiniteProducts D
        F : CategoryTheory.Functor C D
        x✝ : C
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
      -/
    · exact toUnit_unique _ _
      /-
        🎉 no goals
      -/



lemma η_of_chosenFiniteProducts : η F = terminalComparison F := rfl


lemma δ_of_chosenFiniteProducts (X Y : C) : δ F X Y = prodComparison F X Y := rfl


instance : IsIso (η F) :=
  terminalComparison_isIso_of_preservesLimits F


instance (A B : C) : IsIso (δ F A B) :=
  isIso_prodComparison_of_preservesLimit_pair F A B


/-- If `F : C ⥤ D` is a functor between categories with chosen finite products
that preserves finite products, then it is a monoidal functor. -/
noncomputable def monoidalOfChosenFiniteProducts : F.Monoidal :=
  Functor.Monoidal.ofOplaxMonoidal F


