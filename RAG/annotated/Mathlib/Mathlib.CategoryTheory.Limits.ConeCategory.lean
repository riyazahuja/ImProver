/-- Given a cone `c` over `F`, we can interpret the legs of `c` as structured arrows
    `c.pt ⟶ F.obj -`. -/
@[simps]
def Cone.toStructuredArrow {F : J ⥤ C} (c : Cone F) : J ⥤ StructuredArrow c.pt F where
  obj j := StructuredArrow.mk (c.π.app j)
           /-
             J : Type u₁
             inst✝³ : CategoryTheory.Category.{v₁, u₁} J
             K : Type u₂
             inst✝² : CategoryTheory.Category.{v₂, u₂} K
             C : Type u₃
             inst✝¹ : CategoryTheory.Category.{v₃, u₃} C
             D : Type u₄
             inst✝ : CategoryTheory.Category.{v₄, u₄} D
             F : CategoryTheory.Functor J C
             c : CategoryTheory.Limits.Cone F
             X✝ Y✝ : J
             f : Quiver.Hom X✝ Y✝
             ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun j => CategoryTheory.StructuredA …
           -/
  map f := StructuredArrow.homMk f
           /-
             🎉 no goals
           -/


/-- If `F` has a limit, then the limit projections can be interpreted as structured arrows
    `limit F ⟶ F.obj -`. -/
@[simps]
noncomputable def limit.toStructuredArrow (F : J ⥤ C) [HasLimit F] :
    J ⥤ StructuredArrow (limit F) F where
  obj j := StructuredArrow.mk (limit.π F j)
           /-
             J : Type u₁
             inst✝⁴ : CategoryTheory.Category.{v₁, u₁} J
             K : Type u₂
             inst✝³ : CategoryTheory.Category.{v₂, u₂} K
             C : Type u₃
             inst✝² : CategoryTheory.Category.{v₃, u₃} C
             D : Type u₄
             inst✝¹ : CategoryTheory.Category.{v₄, u₄} D
             F : CategoryTheory.Functor J C
             inst✝ : CategoryTheory.Limits.HasLimit F
             X✝ Y✝ : J
             f : Quiver.Hom X✝ Y✝
             ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun j => CategoryTheory.StructuredA …
           -/
  map f := StructuredArrow.homMk f
           /-
             🎉 no goals
           -/


/-- `Cone.toStructuredArrow` can be expressed in terms of `Functor.toStructuredArrow`. -/
def Cone.toStructuredArrowIsoToStructuredArrow {F : J ⥤ C} (c : Cone F) :
                                                                     /-
                                                                       J : Type u₁
                                                                       inst✝³ : CategoryTheory.Category.{v₁, u₁} J
                                                                       K : Type u₂
                                                                       inst✝² : CategoryTheory.Category.{v₂, u₂} K
                                                                       C : Type u₃
                                                                       inst✝¹ : CategoryTheory.Category.{v₃, u₃} C
                                                                       D : Type u₄
                                                                       inst✝ : CategoryTheory.Category.{v₄, u₄} D
                                                                       F : CategoryTheory.Functor J C
                                                                       c : CategoryTheory.Limits.Cone F
                                                                       ⊢ ∀ {Y Z : J} (g : Quiver.Hom Y Z), Eq (CategoryTheory.CategoryStruct.comp (c. …
                                                                     -/
    c.toStructuredArrow ≅ (𝟭 J).toStructuredArrow c.pt F c.π.app (by simp) :=
                                                                     /-
                                                                       🎉 no goals
                                                                     -/
  Iso.refl _


/-- `Functor.toStructuredArrow` can be expressed in terms of `Cone.toStructuredArrow`. -/
def _root_.CategoryTheory.Functor.toStructuredArrowIsoToStructuredArrow (G : J ⥤ K) (X : C)
    (F : K ⥤ C) (f : (Y : J) → X ⟶ F.obj (G.obj Y))
    (h : ∀ {Y Z : J} (g : Y ⟶ Z), f Y ≫ F.map (G.map g) = f Z) :
    G.toStructuredArrow X F f h ≅
                        /-
                          J : Type u₁
                          inst✝³ : CategoryTheory.Category.{v₁, u₁} J
                          K : Type u₂
                          inst✝² : CategoryTheory.Category.{v₂, u₂} K
                          C : Type u₃
                          inst✝¹ : CategoryTheory.Category.{v₃, u₃} C
                          D : Type u₄
                          inst✝ : CategoryTheory.Category.{v₄, u₄} D
                          G : CategoryTheory.Functor J K
                          X : C
                          F : CategoryTheory.Functor K C
                          f : (Y : J) → Quiver.Hom X (F.obj (G.obj Y))
                          h : ∀ {Y Z : J} (g : Quiver.Hom Y Z), Eq (CategoryTheory.CategoryStruct.comp ( …
                          ⊢ ∀ ⦃X_1 Y : J⦄ (f_1 : Quiver.Hom X_1 Y), Eq (CategoryTheory.CategoryStruct.co …
                        -/
      (Cone.mk X ⟨f, by simp [h]⟩).toStructuredArrow ⋙ StructuredArrow.pre _ _ _ :=
                        /-
                          🎉 no goals
                        -/
  Iso.refl _


/-- Interpreting the legs of a cone as a structured arrow and then forgetting the arrow again does
    nothing. -/
@[simps!]
def Cone.toStructuredArrowCompProj {F : J ⥤ C} (c : Cone F) :
    c.toStructuredArrow ⋙ StructuredArrow.proj _ _ ≅ 𝟭 J :=
  Iso.refl _


@[simp]
lemma Cone.toStructuredArrow_comp_proj {F : J ⥤ C} (c : Cone F) :
    c.toStructuredArrow ⋙ StructuredArrow.proj _ _ = 𝟭 J :=
  rfl


/-- Interpreting the legs of a cone as a structured arrow, interpreting this arrow as an arrow over
    the cone point, and finally forgetting the arrow is the same as just applying the functor the
    cone was over. -/
@[simps!]
def Cone.toStructuredArrowCompToUnderCompForget {F : J ⥤ C} (c : Cone F) :
    c.toStructuredArrow ⋙ StructuredArrow.toUnder _ _ ⋙ Under.forget _ ≅ F :=
  Iso.refl _


@[simp]
lemma Cone.toStructuredArrow_comp_toUnder_comp_forget {F : J ⥤ C} (c : Cone F) :
    c.toStructuredArrow ⋙ StructuredArrow.toUnder _ _ ⋙ Under.forget _ = F :=
  rfl


/-- A cone `c` on `F : J ⥤ C` lifts to a cone in `Over c.pt` with cone point `𝟙 c.pt`. -/
@[simps]
def Cone.toUnder {F : J ⥤ C} (c : Cone F) :
    Cone (c.toStructuredArrow ⋙ StructuredArrow.toUnder _ _) where
  pt := Under.mk (𝟙 c.pt)
                                                     /-
                                                       J : Type u₁
                                                       inst✝³ : CategoryTheory.Category.{v₁, u₁} J
                                                       K : Type u₂
                                                       inst✝² : CategoryTheory.Category.{v₂, u₂} K
                                                       C : Type u₃
                                                       inst✝¹ : CategoryTheory.Category.{v₃, u₃} C
                                                       D : Type u₄
                                                       inst✝ : CategoryTheory.Category.{v₄, u₄} D
                                                       F : CategoryTheory.Functor J C
                                                       c : CategoryTheory.Limits.Cone F
                                                       j : J
                                                       ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Functor.const J).ob …
                                                     -/
  π := { app := fun j => Under.homMk (c.π.app j) (by simp) }
                                                     /-
                                                       🎉 no goals
                                                     -/


/-- The limit cone for `F : J ⥤ C` lifts to a cocone in `Under (limit F)` with cone point
    `𝟙 (limit F)`. This is automatically also a limit cone. -/
noncomputable def limit.toUnder (F : J ⥤ C) [HasLimit F] :
    Cone (limit.toStructuredArrow F ⋙ StructuredArrow.toUnder _ _) where
  pt := Under.mk (𝟙 (limit F))
                                                       /-
                                                         J : Type u₁
                                                         inst✝⁴ : CategoryTheory.Category.{v₁, u₁} J
                                                         K : Type u₂
                                                         inst✝³ : CategoryTheory.Category.{v₂, u₂} K
                                                         C : Type u₃
                                                         inst✝² : CategoryTheory.Category.{v₃, u₃} C
                                                         D : Type u₄
                                                         inst✝¹ : CategoryTheory.Category.{v₄, u₄} D
                                                         F : CategoryTheory.Functor J C
                                                         inst✝ : CategoryTheory.Limits.HasLimit F
                                                         j : J
                                                         ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Functor.const J).ob …
                                                       -/
  π := { app := fun j => Under.homMk (limit.π F j) (by simp) }
                                                       /-
                                                         🎉 no goals
                                                       -/


/-- `c.toUnder` is a lift of `c` under the forgetful functor. -/
@[simps!]
def Cone.mapConeToUnder {F : J ⥤ C} (c : Cone F) : (Under.forget c.pt).mapCone c.toUnder ≅ c :=
  Iso.refl _


/-- Given a diagram of `StructuredArrow X F`s, we may obtain a cone with cone point `X`. -/
@[simps!]
def Cone.fromStructuredArrow (F : C ⥤ D) {X : D} (G : J ⥤ StructuredArrow X F) :
    Cone (G ⋙ StructuredArrow.proj X F ⋙ F) where
  π := { app := fun j => (G.obj j).hom }


/-- Given a cone `c : Cone K` and a map `f : X ⟶ F.obj c.X`, we can construct a cone of structured
arrows over `X` with `f` as the cone point.
-/
@[simps]
def Cone.toStructuredArrowCone {K : J ⥤ C} (c : Cone K) (F : C ⥤ D) {X : D} (f : X ⟶ F.obj c.pt) :
    Cone ((F.mapCone c).toStructuredArrow ⋙ StructuredArrow.map f ⋙ StructuredArrow.pre _ K F) where
  pt := StructuredArrow.mk f
  π := { app := fun j => StructuredArrow.homMk (c.π.app j) rfl }


/-- Construct an object of the category `(Δ ↓ F)` from a cone on `F`. This is part of an
    equivalence, see `Cone.equivCostructuredArrow`. -/
@[simps]
def Cone.toCostructuredArrow (F : J ⥤ C) : Cone F ⥤ CostructuredArrow (const J) F where
  obj c := CostructuredArrow.mk c.π
           /-
             J : Type u₁
             inst✝³ : CategoryTheory.Category.{v₁, u₁} J
             K : Type u₂
             inst✝² : CategoryTheory.Category.{v₂, u₂} K
             C : Type u₃
             inst✝¹ : CategoryTheory.Category.{v₃, u₃} C
             D : Type u₄
             inst✝ : CategoryTheory.Category.{v₄, u₄} D
             F : CategoryTheory.Functor J C
             X✝ Y✝ : CategoryTheory.Limits.Cone F
             f : Quiver.Hom X✝ Y✝
             ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Functor.const J).map …
           -/
  map f := CostructuredArrow.homMk f.hom
           /-
             🎉 no goals
           -/


/-- Construct a cone on `F` from an object of the category `(Δ ↓ F)`. This is part of an
    equivalence, see `Cone.equivCostructuredArrow`. -/
@[simps]
def Cone.fromCostructuredArrow (F : J ⥤ C) : CostructuredArrow (const J) F ⥤ Cone F where
  obj c := ⟨c.left, c.hom⟩
  map f :=
    { hom := f.left
      w := fun j => by
        /-
          J : Type u₁
          inst✝³ : CategoryTheory.Category.{v₁, u₁} J
          K : Type u₂
          inst✝² : CategoryTheory.Category.{v₂, u₂} K
          C : Type u₃
          inst✝¹ : CategoryTheory.Category.{v₃, u₃} C
          D : Type u₄
          inst✝ : CategoryTheory.Category.{v₄, u₄} D
          F : CategoryTheory.Functor J C
          X✝ Y✝ : CategoryTheory.CostructuredArrow (CategoryTheory.Functor.const J) F
          f : Quiver.Hom X✝ Y✝
          j : J
          ⊢ Eq (CategoryTheory.CategoryStruct.comp f.left (((fun c => { pt := c.left, π  …
        -/
        convert congr_fun (congr_arg NatTrans.app f.w) j
        /-
          case h.e'_3.h.h.e'_7.h
          J : Type u₁
          inst✝³ : CategoryTheory.Category.{v₁, u₁} J
          K : Type u₂
          inst✝² : CategoryTheory.Category.{v₂, u₂} K
          C : Type u₃
          inst✝¹ : CategoryTheory.Category.{v₃, u₃} C
          D : Type u₄
          inst✝ : CategoryTheory.Category.{v₄, u₄} D
          F : CategoryTheory.Functor J C
          X✝ Y✝ : CategoryTheory.CostructuredArrow (CategoryTheory.Functor.const J) F
          f : Quiver.Hom X✝ Y✝
          j : J
          e_1✝ : Eq (Quiver.Hom ((fun c => { pt := c.left, π := c.hom }) X✝).pt (F.obj j …
          e_6✝ : Eq F ((CategoryTheory.Functor.fromPUnit F).obj Y✝.right)
          ⊢ Eq ((fun c => { pt := c.left, π := c.hom }) X✝).π (CategoryTheory.CategorySt …
        -/
        dsimp
        /-
          case h.e'_3.h.h.e'_7.h
          J : Type u₁
          inst✝³ : CategoryTheory.Category.{v₁, u₁} J
          K : Type u₂
          inst✝² : CategoryTheory.Category.{v₂, u₂} K
          C : Type u₃
          inst✝¹ : CategoryTheory.Category.{v₃, u₃} C
          D : Type u₄
          inst✝ : CategoryTheory.Category.{v₄, u₄} D
          F : CategoryTheory.Functor J C
          X✝ Y✝ : CategoryTheory.CostructuredArrow (CategoryTheory.Functor.const J) F
          f : Quiver.Hom X✝ Y✝
          j : J
          e_1✝ : Eq (Quiver.Hom ((fun c => { pt := c.left, π := c.hom }) X✝).pt (F.obj j …
          e_6✝ : Eq F ((CategoryTheory.Functor.fromPUnit F).obj Y✝.right)
          ⊢ Eq X✝.hom (CategoryTheory.CategoryStruct.comp X✝.hom (CategoryTheory.Categor …
        -/
        simp }
        /-
          🎉 no goals
        -/


/-- The category of cones on `F` is just the comma category `(Δ ↓ F)`, where `Δ` is the constant
    functor. -/
@[simps]
def Cone.equivCostructuredArrow (F : J ⥤ C) : Cone F ≌ CostructuredArrow (const J) F where
  functor := Cone.toCostructuredArrow F
  inverse := Cone.fromCostructuredArrow F
             /-
               J : Type u₁
               inst✝³ : CategoryTheory.Category.{v₁, u₁} J
               K : Type u₂
               inst✝² : CategoryTheory.Category.{v₂, u₂} K
               C : Type u₃
               inst✝¹ : CategoryTheory.Category.{v₃, u₃} C
               D : Type u₄
               inst✝ : CategoryTheory.Category.{v₄, u₄} D
               F : CategoryTheory.Functor J C
               ⊢ ∀ {X Y : CategoryTheory.Limits.Cone F} (f : Quiver.Hom X Y), Eq (CategoryThe …
             -/
  unitIso := NatIso.ofComponents Cones.eta
             /-
               🎉 no goals
             -/
               /-
                 J : Type u₁
                 inst✝³ : CategoryTheory.Category.{v₁, u₁} J
                 K : Type u₂
                 inst✝² : CategoryTheory.Category.{v₂, u₂} K
                 C : Type u₃
                 inst✝¹ : CategoryTheory.Category.{v₃, u₃} C
                 D : Type u₄
                 inst✝ : CategoryTheory.Category.{v₄, u₄} D
                 F : CategoryTheory.Functor J C
                 ⊢ ∀ {X Y : CategoryTheory.CostructuredArrow (CategoryTheory.Functor.const J) F …
               -/
  counitIso := NatIso.ofComponents fun _ => (CostructuredArrow.eta _).symm
               /-
                 🎉 no goals
               -/


/-- A cone is a limit cone iff it is terminal. -/
def Cone.isLimitEquivIsTerminal {F : J ⥤ C} (c : Cone F) : IsLimit c ≃ IsTerminal c :=
  IsLimit.isoUniqueConeMorphism.toEquiv.trans
    { toFun := fun _ => IsTerminal.ofUnique _
      invFun := fun h s => ⟨⟨IsTerminal.from h s⟩, fun a => IsTerminal.hom_ext h a _⟩
                     /-
                       J : Type u₁
                       inst✝³ : CategoryTheory.Category.{v₁, u₁} J
                       K : Type u₂
                       inst✝² : CategoryTheory.Category.{v₂, u₂} K
                       C : Type u₃
                       inst✝¹ : CategoryTheory.Category.{v₃, u₃} C
                       D : Type u₄
                       inst✝ : CategoryTheory.Category.{v₄, u₄} D
                       F : CategoryTheory.Functor J C
                       c : CategoryTheory.Limits.Cone F
                       ⊢ Function.LeftInverse (fun h s => { default := h.from s, uniq := ⋯ }) fun x = …
                     -/
      left_inv := by aesop_cat
                     /-
                       🎉 no goals
                     -/
                      /-
                        J : Type u₁
                        inst✝³ : CategoryTheory.Category.{v₁, u₁} J
                        K : Type u₂
                        inst✝² : CategoryTheory.Category.{v₂, u₂} K
                        C : Type u₃
                        inst✝¹ : CategoryTheory.Category.{v₃, u₃} C
                        D : Type u₄
                        inst✝ : CategoryTheory.Category.{v₄, u₄} D
                        F : CategoryTheory.Functor J C
                        c : CategoryTheory.Limits.Cone F
                        ⊢ Function.RightInverse (fun h s => { default := h.from s, uniq := ⋯ }) fun x  …
                      -/
      right_inv := by aesop_cat }
                      /-
                        🎉 no goals
                      -/


theorem hasLimit_iff_hasTerminal_cone (F : J ⥤ C) : HasLimit F ↔ HasTerminal (Cone F) :=
  ⟨fun _ => (Cone.isLimitEquivIsTerminal _ (limit.isLimit F)).hasTerminal, fun h =>
    haveI : HasTerminal (Cone F) := h
    ⟨⟨⟨⊤_ _, (Cone.isLimitEquivIsTerminal _).symm terminalIsTerminal⟩⟩⟩⟩


theorem hasLimitsOfShape_iff_isLeftAdjoint_const :
    HasLimitsOfShape J C ↔ IsLeftAdjoint (const J : C ⥤ _) :=
  calc
    HasLimitsOfShape J C ↔ ∀ F : J ⥤ C, HasLimit F :=
                                      /-
                                        J : Type u₁
                                        inst✝¹ : CategoryTheory.Category.{v₁, u₁} J
                                        C : Type u₃
                                        inst✝ : CategoryTheory.Category.{v₃, u₃} C
                                        h : ∀ (F : CategoryTheory.Functor J C), CategoryTheory.Limits.HasLimit F
                                        ⊢ ∀ (F : CategoryTheory.Functor J C), CategoryTheory.Limits.HasLimit F
                                      -/
      ⟨fun h => h.has_limit, fun h => HasLimitsOfShape.mk⟩
                                      /-
                                        🎉 no goals
                                      -/
    _ ↔ ∀ F : J ⥤ C, HasTerminal (Cone F) := forall_congr' hasLimit_iff_hasTerminal_cone
    _ ↔ ∀ F : J ⥤ C, HasTerminal (CostructuredArrow (const J) F) :=
      (forall_congr' fun F => (Cone.equivCostructuredArrow F).hasTerminal_iff)
    _ ↔ (IsLeftAdjoint (const J : C ⥤ _)) :=
      isLeftAdjoint_iff_hasTerminal_costructuredArrow.symm


theorem IsLimit.liftConeMorphism_eq_isTerminal_from {F : J ⥤ C} {c : Cone F} (hc : IsLimit c)
    (s : Cone F) : hc.liftConeMorphism s = IsTerminal.from (Cone.isLimitEquivIsTerminal _ hc) _ :=
  rfl


theorem IsTerminal.from_eq_liftConeMorphism {F : J ⥤ C} {c : Cone F} (hc : IsTerminal c)
    (s : Cone F) :
    IsTerminal.from hc s = ((Cone.isLimitEquivIsTerminal _).symm hc).liftConeMorphism s :=
  (IsLimit.liftConeMorphism_eq_isTerminal_from (c.isLimitEquivIsTerminal.symm hc) s).symm


/-- If `G : Cone F ⥤ Cone F'` preserves terminal objects, it preserves limit cones. -/
noncomputable def IsLimit.ofPreservesConeTerminal {F : J ⥤ C} {F' : K ⥤ D} (G : Cone F ⥤ Cone F')
    [PreservesLimit (Functor.empty.{0} _) G] {c : Cone F} (hc : IsLimit c) : IsLimit (G.obj c) :=
  (Cone.isLimitEquivIsTerminal _).symm <| (Cone.isLimitEquivIsTerminal _ hc).isTerminalObj _ _


/-- If `G : Cone F ⥤ Cone F'` reflects terminal objects, it reflects limit cones. -/
noncomputable def IsLimit.ofReflectsConeTerminal {F : J ⥤ C} {F' : K ⥤ D} (G : Cone F ⥤ Cone F')
    [ReflectsLimit (Functor.empty.{0} _) G] {c : Cone F} (hc : IsLimit (G.obj c)) : IsLimit c :=
  (Cone.isLimitEquivIsTerminal _).symm <| (Cone.isLimitEquivIsTerminal _ hc).isTerminalOfObj _ _


/-- Given a cocone `c` over `F`, we can interpret the legs of `c` as costructured arrows
    `F.obj - ⟶ c.pt`. -/
@[simps]
def Cocone.toCostructuredArrow {F : J ⥤ C} (c : Cocone F) : J ⥤ CostructuredArrow F c.pt where
  obj j := CostructuredArrow.mk (c.ι.app j)
           /-
             J : Type u₁
             inst✝³ : CategoryTheory.Category.{v₁, u₁} J
             K : Type u₂
             inst✝² : CategoryTheory.Category.{v₂, u₂} K
             C : Type u₃
             inst✝¹ : CategoryTheory.Category.{v₃, u₃} C
             D : Type u₄
             inst✝ : CategoryTheory.Category.{v₄, u₄} D
             F : CategoryTheory.Functor J C
             c : CategoryTheory.Limits.Cocone F
             X✝ Y✝ : J
             f : Quiver.Hom X✝ Y✝
             ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map f) ((fun j => CategoryTheory.C …
           -/
  map f := CostructuredArrow.homMk f
           /-
             🎉 no goals
           -/


/-- If `F` has a colimit, then the colimit inclusions can be interpreted as costructured arrows
    `F.obj - ⟶ colimit F`. -/
@[simps]
noncomputable def colimit.toCostructuredArrow (F : J ⥤ C) [HasColimit F] :
    J ⥤ CostructuredArrow F (colimit F) where
  obj j := CostructuredArrow.mk (colimit.ι F j)
           /-
             J : Type u₁
             inst✝⁴ : CategoryTheory.Category.{v₁, u₁} J
             K : Type u₂
             inst✝³ : CategoryTheory.Category.{v₂, u₂} K
             C : Type u₃
             inst✝² : CategoryTheory.Category.{v₃, u₃} C
             D : Type u₄
             inst✝¹ : CategoryTheory.Category.{v₄, u₄} D
             F : CategoryTheory.Functor J C
             inst✝ : CategoryTheory.Limits.HasColimit F
             X✝ Y✝ : J
             f : Quiver.Hom X✝ Y✝
             ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map f) ((fun j => CategoryTheory.C …
           -/
  map f := CostructuredArrow.homMk f
           /-
             🎉 no goals
           -/


/-- `Cocone.toCostructuredArrow` can be expressed in terms of `Functor.toCostructuredArrow`. -/
def Cocone.toCostructuredArrowIsoToCostructuredArrow {F : J ⥤ C} (c : Cocone F) :
                                                                         /-
                                                                           J : Type u₁
                                                                           inst✝³ : CategoryTheory.Category.{v₁, u₁} J
                                                                           K : Type u₂
                                                                           inst✝² : CategoryTheory.Category.{v₂, u₂} K
                                                                           C : Type u₃
                                                                           inst✝¹ : CategoryTheory.Category.{v₃, u₃} C
                                                                           D : Type u₄
                                                                           inst✝ : CategoryTheory.Category.{v₄, u₄} D
                                                                           F : CategoryTheory.Functor J C
                                                                           c : CategoryTheory.Limits.Cocone F
                                                                           ⊢ ∀ {Y Z : J} (g : Quiver.Hom Y Z), Eq (CategoryTheory.CategoryStruct.comp (F. …
                                                                         -/
    c.toCostructuredArrow ≅ (𝟭 J).toCostructuredArrow F c.pt c.ι.app (by simp) :=
                                                                         /-
                                                                           🎉 no goals
                                                                         -/
  Iso.refl _


/-- `Functor.toCostructuredArrow` can be expressed in terms of `Cocone.toCostructuredArrow`. -/
def _root_.CategoryTheory.Functor.toCostructuredArrowIsoToCostructuredArrow (G : J ⥤ K)
    (F : K ⥤ C) (X : C) (f : (Y : J) → F.obj (G.obj Y) ⟶ X)
    (h : ∀ {Y Z : J} (g : Y ⟶ Z), F.map (G.map g) ≫ f Z = f Y) :
    G.toCostructuredArrow F X f h ≅
                          /-
                            J : Type u₁
                            inst✝³ : CategoryTheory.Category.{v₁, u₁} J
                            K : Type u₂
                            inst✝² : CategoryTheory.Category.{v₂, u₂} K
                            C : Type u₃
                            inst✝¹ : CategoryTheory.Category.{v₃, u₃} C
                            D : Type u₄
                            inst✝ : CategoryTheory.Category.{v₄, u₄} D
                            G : CategoryTheory.Functor J K
                            F : CategoryTheory.Functor K C
                            X : C
                            f : (Y : J) → Quiver.Hom (F.obj (G.obj Y)) X
                            h : ∀ {Y Z : J} (g : Quiver.Hom Y Z), Eq (CategoryTheory.CategoryStruct.comp ( …
                            ⊢ ∀ ⦃X_1 Y : J⦄ (f_1 : Quiver.Hom X_1 Y), Eq (CategoryTheory.CategoryStruct.co …
                          -/
      (Cocone.mk X ⟨f, by simp [h]⟩).toCostructuredArrow ⋙ CostructuredArrow.pre _ _ _ :=
                          /-
                            🎉 no goals
                          -/
  Iso.refl _


/-- Interpreting the legs of a cocone as a costructured arrow and then forgetting the arrow again
    does nothing. -/
@[simps!]
def Cocone.toCostructuredArrowCompProj {F : J ⥤ C} (c : Cocone F) :
    c.toCostructuredArrow ⋙ CostructuredArrow.proj _ _ ≅ 𝟭 J :=
  Iso.refl _


@[simp]
lemma Cocone.toCostructuredArrow_comp_proj {F : J ⥤ C} (c : Cocone F) :
    c.toCostructuredArrow ⋙ CostructuredArrow.proj _ _ = 𝟭 J :=
  rfl


/-- Interpreting the legs of a cocone as a costructured arrow, interpreting this arrow as an arrow
    over the cocone point, and finally forgetting the arrow is the same as just applying the
    functor the cocone was over. -/
@[simps!]
def Cocone.toCostructuredArrowCompToOverCompForget {F : J ⥤ C} (c : Cocone F) :
    c.toCostructuredArrow ⋙ CostructuredArrow.toOver _ _ ⋙ Over.forget _ ≅ F :=
  Iso.refl _


@[simp]
lemma Cocone.toCostructuredArrow_comp_toOver_comp_forget {F : J ⥤ C} (c : Cocone F) :
    c.toCostructuredArrow ⋙ CostructuredArrow.toOver _ _ ⋙ Over.forget _ = F :=
  rfl


/-- A cocone `c` on `F : J ⥤ C` lifts to a cocone in `Over c.pt` with cone point `𝟙 c.pt`. -/
@[simps]
def Cocone.toOver {F : J ⥤ C} (c : Cocone F) :
    Cocone (c.toCostructuredArrow ⋙ CostructuredArrow.toOver _ _) where
  pt := Over.mk (𝟙 c.pt)
                                                    /-
                                                      J : Type u₁
                                                      inst✝³ : CategoryTheory.Category.{v₁, u₁} J
                                                      K : Type u₂
                                                      inst✝² : CategoryTheory.Category.{v₂, u₂} K
                                                      C : Type u₃
                                                      inst✝¹ : CategoryTheory.Category.{v₃, u₃} C
                                                      D : Type u₄
                                                      inst✝ : CategoryTheory.Category.{v₄, u₄} D
                                                      F : CategoryTheory.Functor J C
                                                      c : CategoryTheory.Limits.Cocone F
                                                      j : J
                                                      ⊢ Eq (CategoryTheory.CategoryStruct.comp (c.ι.app j) (((CategoryTheory.Functor …
                                                    -/
  ι := { app := fun j => Over.homMk (c.ι.app j) (by simp) }
                                                    /-
                                                      🎉 no goals
                                                    -/


/-- The colimit cocone for `F : J ⥤ C` lifts to a cocone in `Over (colimit F)` with cone point
    `𝟙 (colimit F)`. This is automatically also a colimit cocone. -/
@[simps]
noncomputable def colimit.toOver (F : J ⥤ C) [HasColimit F] :
    Cocone (colimit.toCostructuredArrow F ⋙ CostructuredArrow.toOver _ _) where
  pt := Over.mk (𝟙 (colimit F))
                                                        /-
                                                          J : Type u₁
                                                          inst✝⁴ : CategoryTheory.Category.{v₁, u₁} J
                                                          K : Type u₂
                                                          inst✝³ : CategoryTheory.Category.{v₂, u₂} K
                                                          C : Type u₃
                                                          inst✝² : CategoryTheory.Category.{v₃, u₃} C
                                                          D : Type u₄
                                                          inst✝¹ : CategoryTheory.Category.{v₄, u₄} D
                                                          F : CategoryTheory.Functor J C
                                                          inst✝ : CategoryTheory.Limits.HasColimit F
                                                          j : J
                                                          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.ι F j) …
                                                        -/
  ι := { app := fun j => Over.homMk (colimit.ι F j) (by simp) }
                                                        /-
                                                          🎉 no goals
                                                        -/


/-- `c.toOver` is a lift of `c` under the forgetful functor. -/
@[simps!]
def Cocone.mapCoconeToOver {F : J ⥤ C} (c : Cocone F) : (Over.forget c.pt).mapCocone c.toOver ≅ c :=
  Iso.refl _


/-- Given a diagram `CostructuredArrow F X`s, we may obtain a cocone with cone point `X`. -/
@[simps!]
def Cocone.fromCostructuredArrow (F : C ⥤ D) {X : D} (G : J ⥤ CostructuredArrow F X) :
    Cocone (G ⋙ CostructuredArrow.proj F X ⋙ F) where
  ι := { app := fun j => (G.obj j).hom }


/-- Given a cocone `c : Cocone K` and a map `f : F.obj c.X ⟶ X`, we can construct a cocone of
    costructured arrows over `X` with `f` as the cone point. -/
@[simps]
def Cocone.toCostructuredArrowCocone {K : J ⥤ C} (c : Cocone K) (F : C ⥤ D) {X : D}
    (f : F.obj c.pt ⟶ X) : Cocone ((F.mapCocone c).toCostructuredArrow ⋙
      CostructuredArrow.map f ⋙ CostructuredArrow.pre _ _ _) where
  pt := CostructuredArrow.mk f
  ι := { app := fun j => CostructuredArrow.homMk (c.ι.app j) rfl }


/-- Construct an object of the category `(F ↓ Δ)` from a cocone on `F`. This is part of an
    equivalence, see `Cocone.equivStructuredArrow`. -/
@[simps]
def Cocone.toStructuredArrow (F : J ⥤ C) : Cocone F ⥤ StructuredArrow F (const J) where
  obj c := StructuredArrow.mk c.ι
           /-
             J : Type u₁
             inst✝³ : CategoryTheory.Category.{v₁, u₁} J
             K : Type u₂
             inst✝² : CategoryTheory.Category.{v₂, u₂} K
             C : Type u₃
             inst✝¹ : CategoryTheory.Category.{v₃, u₃} C
             D : Type u₄
             inst✝ : CategoryTheory.Category.{v₄, u₄} D
             F : CategoryTheory.Functor J C
             X✝ Y✝ : CategoryTheory.Limits.Cocone F
             f : Quiver.Hom X✝ Y✝
             ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun c => CategoryTheory.StructuredA …
           -/
  map f := StructuredArrow.homMk f.hom
           /-
             🎉 no goals
           -/


/-- Construct a cocone on `F` from an object of the category `(F ↓ Δ)`. This is part of an
    equivalence, see `Cocone.equivStructuredArrow`. -/
@[simps]
def Cocone.fromStructuredArrow (F : J ⥤ C) : StructuredArrow F (const J) ⥤ Cocone F where
  obj c := ⟨c.right, c.hom⟩
  map f :=
    { hom := f.right
      w := fun j => by
        /-
          J : Type u₁
          inst✝³ : CategoryTheory.Category.{v₁, u₁} J
          K : Type u₂
          inst✝² : CategoryTheory.Category.{v₂, u₂} K
          C : Type u₃
          inst✝¹ : CategoryTheory.Category.{v₃, u₃} C
          D : Type u₄
          inst✝ : CategoryTheory.Category.{v₄, u₄} D
          F : CategoryTheory.Functor J C
          X✝ Y✝ : CategoryTheory.StructuredArrow F (CategoryTheory.Functor.const J)
          f : Quiver.Hom X✝ Y✝
          j : J
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (((fun c => { pt := c.right, ι := c.h …
        -/
        convert (congr_fun (congr_arg NatTrans.app f.w) j).symm
        /-
          case h.e'_3.h.h.e'_7.h
          J : Type u₁
          inst✝³ : CategoryTheory.Category.{v₁, u₁} J
          K : Type u₂
          inst✝² : CategoryTheory.Category.{v₂, u₂} K
          C : Type u₃
          inst✝¹ : CategoryTheory.Category.{v₃, u₃} C
          D : Type u₄
          inst✝ : CategoryTheory.Category.{v₄, u₄} D
          F : CategoryTheory.Functor J C
          X✝ Y✝ : CategoryTheory.StructuredArrow F (CategoryTheory.Functor.const J)
          f : Quiver.Hom X✝ Y✝
          j : J
          e_1✝ : Eq (Quiver.Hom (F.obj j) ((fun c => { pt := c.right, ι := c.hom }) Y✝). …
          e_5✝ : Eq F ((CategoryTheory.Functor.fromPUnit F).obj X✝.left)
          ⊢ Eq ((fun c => { pt := c.right, ι := c.hom }) Y✝).ι (CategoryTheory.CategoryS …
        -/
        dsimp
        /-
          case h.e'_3.h.h.e'_7.h
          J : Type u₁
          inst✝³ : CategoryTheory.Category.{v₁, u₁} J
          K : Type u₂
          inst✝² : CategoryTheory.Category.{v₂, u₂} K
          C : Type u₃
          inst✝¹ : CategoryTheory.Category.{v₃, u₃} C
          D : Type u₄
          inst✝ : CategoryTheory.Category.{v₄, u₄} D
          F : CategoryTheory.Functor J C
          X✝ Y✝ : CategoryTheory.StructuredArrow F (CategoryTheory.Functor.const J)
          f : Quiver.Hom X✝ Y✝
          j : J
          e_1✝ : Eq (Quiver.Hom (F.obj j) ((fun c => { pt := c.right, ι := c.hom }) Y✝). …
          e_5✝ : Eq F ((CategoryTheory.Functor.fromPUnit F).obj X✝.left)
          ⊢ Eq Y✝.hom (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct …
        -/
        simp }
        /-
          🎉 no goals
        -/


/-- The category of cocones on `F` is just the comma category `(F ↓ Δ)`, where `Δ` is the constant
    functor. -/
@[simps]
def Cocone.equivStructuredArrow (F : J ⥤ C) : Cocone F ≌ StructuredArrow F (const J) where
  functor := Cocone.toStructuredArrow F
  inverse := Cocone.fromStructuredArrow F
             /-
               J : Type u₁
               inst✝³ : CategoryTheory.Category.{v₁, u₁} J
               K : Type u₂
               inst✝² : CategoryTheory.Category.{v₂, u₂} K
               C : Type u₃
               inst✝¹ : CategoryTheory.Category.{v₃, u₃} C
               D : Type u₄
               inst✝ : CategoryTheory.Category.{v₄, u₄} D
               F : CategoryTheory.Functor J C
               ⊢ ∀ {X Y : CategoryTheory.Limits.Cocone F} (f : Quiver.Hom X Y), Eq (CategoryT …
             -/
  unitIso := NatIso.ofComponents Cocones.eta
             /-
               🎉 no goals
             -/
               /-
                 J : Type u₁
                 inst✝³ : CategoryTheory.Category.{v₁, u₁} J
                 K : Type u₂
                 inst✝² : CategoryTheory.Category.{v₂, u₂} K
                 C : Type u₃
                 inst✝¹ : CategoryTheory.Category.{v₃, u₃} C
                 D : Type u₄
                 inst✝ : CategoryTheory.Category.{v₄, u₄} D
                 F : CategoryTheory.Functor J C
                 ⊢ ∀ {X Y : CategoryTheory.StructuredArrow F (CategoryTheory.Functor.const J)}  …
               -/
  counitIso := NatIso.ofComponents fun _ => (StructuredArrow.eta _).symm
               /-
                 🎉 no goals
               -/


/-- A cocone is a colimit cocone iff it is initial. -/
def Cocone.isColimitEquivIsInitial {F : J ⥤ C} (c : Cocone F) : IsColimit c ≃ IsInitial c :=
  IsColimit.isoUniqueCoconeMorphism.toEquiv.trans
    { toFun := fun _ => IsInitial.ofUnique _
      invFun := fun h s => ⟨⟨IsInitial.to h s⟩, fun a => IsInitial.hom_ext h a _⟩
                     /-
                       J : Type u₁
                       inst✝³ : CategoryTheory.Category.{v₁, u₁} J
                       K : Type u₂
                       inst✝² : CategoryTheory.Category.{v₂, u₂} K
                       C : Type u₃
                       inst✝¹ : CategoryTheory.Category.{v₃, u₃} C
                       D : Type u₄
                       inst✝ : CategoryTheory.Category.{v₄, u₄} D
                       F : CategoryTheory.Functor J C
                       c : CategoryTheory.Limits.Cocone F
                       ⊢ Function.LeftInverse (fun h s => { default := h.to s, uniq := ⋯ }) fun x =>  …
                     -/
      left_inv := by aesop_cat
                     /-
                       🎉 no goals
                     -/
                      /-
                        J : Type u₁
                        inst✝³ : CategoryTheory.Category.{v₁, u₁} J
                        K : Type u₂
                        inst✝² : CategoryTheory.Category.{v₂, u₂} K
                        C : Type u₃
                        inst✝¹ : CategoryTheory.Category.{v₃, u₃} C
                        D : Type u₄
                        inst✝ : CategoryTheory.Category.{v₄, u₄} D
                        F : CategoryTheory.Functor J C
                        c : CategoryTheory.Limits.Cocone F
                        ⊢ Function.RightInverse (fun h s => { default := h.to s, uniq := ⋯ }) fun x => …
                      -/
      right_inv := by aesop_cat }
                      /-
                        🎉 no goals
                      -/


theorem hasColimit_iff_hasInitial_cocone (F : J ⥤ C) : HasColimit F ↔ HasInitial (Cocone F) :=
  ⟨fun _ => (Cocone.isColimitEquivIsInitial _ (colimit.isColimit F)).hasInitial, fun h =>
    haveI : HasInitial (Cocone F) := h
    ⟨⟨⟨⊥_ _, (Cocone.isColimitEquivIsInitial _).symm initialIsInitial⟩⟩⟩⟩


theorem hasColimitsOfShape_iff_isRightAdjoint_const :
    HasColimitsOfShape J C ↔ IsRightAdjoint (const J : C ⥤ _) :=
  calc
    HasColimitsOfShape J C ↔ ∀ F : J ⥤ C, HasColimit F :=
                                        /-
                                          J : Type u₁
                                          inst✝¹ : CategoryTheory.Category.{v₁, u₁} J
                                          C : Type u₃
                                          inst✝ : CategoryTheory.Category.{v₃, u₃} C
                                          h : ∀ (F : CategoryTheory.Functor J C), CategoryTheory.Limits.HasColimit F
                                          ⊢ ∀ (F : CategoryTheory.Functor J C), CategoryTheory.Limits.HasColimit F
                                        -/
      ⟨fun h => h.has_colimit, fun h => HasColimitsOfShape.mk⟩
                                        /-
                                          🎉 no goals
                                        -/
    _ ↔ ∀ F : J ⥤ C, HasInitial (Cocone F) := forall_congr' hasColimit_iff_hasInitial_cocone
    _ ↔ ∀ F : J ⥤ C, HasInitial (StructuredArrow F (const J)) :=
      (forall_congr' fun F => (Cocone.equivStructuredArrow F).hasInitial_iff)
    _ ↔ (IsRightAdjoint (const J : C ⥤ _)) :=
      isRightAdjoint_iff_hasInitial_structuredArrow.symm


theorem IsColimit.descCoconeMorphism_eq_isInitial_to {F : J ⥤ C} {c : Cocone F} (hc : IsColimit c)
    (s : Cocone F) :
    hc.descCoconeMorphism s = IsInitial.to (Cocone.isColimitEquivIsInitial _ hc) _ :=
  rfl


theorem IsInitial.to_eq_descCoconeMorphism {F : J ⥤ C} {c : Cocone F} (hc : IsInitial c)
    (s : Cocone F) :
    IsInitial.to hc s = ((Cocone.isColimitEquivIsInitial _).symm hc).descCoconeMorphism s :=
  (IsColimit.descCoconeMorphism_eq_isInitial_to (c.isColimitEquivIsInitial.symm hc) s).symm


/-- If `G : Cocone F ⥤ Cocone F'` preserves initial objects, it preserves colimit cocones. -/
noncomputable def IsColimit.ofPreservesCoconeInitial {F : J ⥤ C} {F' : K ⥤ D}
    (G : Cocone F ⥤ Cocone F')
    [PreservesColimit (Functor.empty.{0} _) G] {c : Cocone F} (hc : IsColimit c) :
    IsColimit (G.obj c) :=
  (Cocone.isColimitEquivIsInitial _).symm <| (Cocone.isColimitEquivIsInitial _ hc).isInitialObj _ _


/-- If `G : Cocone F ⥤ Cocone F'` reflects initial objects, it reflects colimit cocones. -/
noncomputable def IsColimit.ofReflectsCoconeInitial {F : J ⥤ C} {F' : K ⥤ D}
    (G : Cocone F ⥤ Cocone F')
    [ReflectsColimit (Functor.empty.{0} _) G] {c : Cocone F} (hc : IsColimit (G.obj c)) :
    IsColimit c :=
  (Cocone.isColimitEquivIsInitial _).symm <|
    (Cocone.isColimitEquivIsInitial _ hc).isInitialOfObj _ _


