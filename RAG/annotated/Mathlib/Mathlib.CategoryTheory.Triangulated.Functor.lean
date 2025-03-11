/-- The functor `Triangle C ⥤ Triangle D` that is induced by a functor `F : C ⥤ D`
which commutes with shift by `ℤ`. -/
@[simps]
def mapTriangle : Triangle C ⥤ Triangle D where
  obj T := Triangle.mk (F.map T.mor₁) (F.map T.mor₂)
    (F.map T.mor₃ ≫ (F.commShiftIso (1 : ℤ)).hom.app T.obj₁)
  map f :=
    { hom₁ := F.map f.hom₁
      hom₂ := F.map f.hom₂
      hom₃ := F.map f.hom₃
                  /-
                    C : Type u_1
                    D : Type u_2
                    E : Type u_3
                    inst✝⁷ : CategoryTheory.Category.{?u.314, u_1} C
                    inst✝⁶ : CategoryTheory.Category.{?u.318, u_2} D
                    inst✝⁵ : CategoryTheory.Category.{?u.322, u_3} E
                    inst✝⁴ : CategoryTheory.HasShift C Int
                    inst✝³ : CategoryTheory.HasShift D Int
                    inst✝² : CategoryTheory.HasShift E Int
                    F : CategoryTheory.Functor C D
                    inst✝¹ : F.CommShift Int
                    G : CategoryTheory.Functor D E
                    inst✝ : G.CommShift Int
                    X✝ Y✝ : CategoryTheory.Pretriangulated.Triangle C
                    f : Quiver.Hom X✝ Y✝
                    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun T => CategoryTheory.Pretriangul …
                  -/
      comm₁ := by dsimp; simp only [← F.map_comp, f.comm₁]
                         /-
                           🎉 no goals
                         -/
                  /-
                    C : Type u_1
                    D : Type u_2
                    E : Type u_3
                    inst✝⁷ : CategoryTheory.Category.{?u.314, u_1} C
                    inst✝⁶ : CategoryTheory.Category.{?u.318, u_2} D
                    inst✝⁵ : CategoryTheory.Category.{?u.322, u_3} E
                    inst✝⁴ : CategoryTheory.HasShift C Int
                    inst✝³ : CategoryTheory.HasShift D Int
                    inst✝² : CategoryTheory.HasShift E Int
                    F : CategoryTheory.Functor C D
                    inst✝¹ : F.CommShift Int
                    G : CategoryTheory.Functor D E
                    inst✝ : G.CommShift Int
                    X✝ Y✝ : CategoryTheory.Pretriangulated.Triangle C
                    f : Quiver.Hom X✝ Y✝
                    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun T => CategoryTheory.Pretriangul …
                  -/
      comm₂ := by dsimp; simp only [← F.map_comp, f.comm₂]
                         /-
                           🎉 no goals
                         -/
      comm₃ := by
        /-
          C : Type u_1
          D : Type u_2
          E : Type u_3
          inst✝⁷ : CategoryTheory.Category.{?u.314, u_1} C
          inst✝⁶ : CategoryTheory.Category.{?u.318, u_2} D
          inst✝⁵ : CategoryTheory.Category.{?u.322, u_3} E
          inst✝⁴ : CategoryTheory.HasShift C Int
          inst✝³ : CategoryTheory.HasShift D Int
          inst✝² : CategoryTheory.HasShift E Int
          F : CategoryTheory.Functor C D
          inst✝¹ : F.CommShift Int
          G : CategoryTheory.Functor D E
          inst✝ : G.CommShift Int
          X✝ Y✝ : CategoryTheory.Pretriangulated.Triangle C
          f : Quiver.Hom X✝ Y✝
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun T => CategoryTheory.Pretriangul …
        -/
        dsimp [Functor.comp]
        simp only [Category.assoc, ← NatTrans.naturality,
          ← F.map_comp_assoc, f.comm₃] }


instance [Faithful F] : Faithful F.mapTriangle where
  map_injective {X Y} f g h := by
    /-
      C : Type u_1
      D : Type u_2
      E : Type u_3
      inst✝⁸ : CategoryTheory.Category.{u_4, u_1} C
      inst✝⁷ : CategoryTheory.Category.{u_5, u_2} D
      inst✝⁶ : CategoryTheory.Category.{?u.9225, u_3} E
      inst✝⁵ : CategoryTheory.HasShift C Int
      inst✝⁴ : CategoryTheory.HasShift D Int
      inst✝³ : CategoryTheory.HasShift E Int
      F : CategoryTheory.Functor C D
      inst✝² : F.CommShift Int
      G : CategoryTheory.Functor D E
      inst✝¹ : G.CommShift Int
      inst✝ : F.Faithful
      X Y : CategoryTheory.Pretriangulated.Triangle C
      f g : Quiver.Hom X Y
      h : Eq (F.mapTriangle.map f) (F.mapTriangle.map g)
      ⊢ Eq f g
    -/
    ext <;> apply F.map_injective
      /-
        case h₁.a
        C : Type u_1
        D : Type u_2
        E : Type u_3
        inst✝⁸ : CategoryTheory.Category.{u_4, u_1} C
        inst✝⁷ : CategoryTheory.Category.{u_5, u_2} D
        inst✝⁶ : CategoryTheory.Category.{?u.9225, u_3} E
        inst✝⁵ : CategoryTheory.HasShift C Int
        inst✝⁴ : CategoryTheory.HasShift D Int
        inst✝³ : CategoryTheory.HasShift E Int
        F : CategoryTheory.Functor C D
        inst✝² : F.CommShift Int
        G : CategoryTheory.Functor D E
        inst✝¹ : G.CommShift Int
        inst✝ : F.Faithful
        X Y : CategoryTheory.Pretriangulated.Triangle C
        f g : Quiver.Hom X Y
        h : Eq (F.mapTriangle.map f) (F.mapTriangle.map g)
        ⊢ Eq (F.map f.hom₁) (F.map g.hom₁)
      -/
    · exact congr_arg TriangleMorphism.hom₁ h
      /-
        🎉 no goals
      -/
      /-
        case h₂.a
        C : Type u_1
        D : Type u_2
        E : Type u_3
        inst✝⁸ : CategoryTheory.Category.{u_4, u_1} C
        inst✝⁷ : CategoryTheory.Category.{u_5, u_2} D
        inst✝⁶ : CategoryTheory.Category.{?u.9225, u_3} E
        inst✝⁵ : CategoryTheory.HasShift C Int
        inst✝⁴ : CategoryTheory.HasShift D Int
        inst✝³ : CategoryTheory.HasShift E Int
        F : CategoryTheory.Functor C D
        inst✝² : F.CommShift Int
        G : CategoryTheory.Functor D E
        inst✝¹ : G.CommShift Int
        inst✝ : F.Faithful
        X Y : CategoryTheory.Pretriangulated.Triangle C
        f g : Quiver.Hom X Y
        h : Eq (F.mapTriangle.map f) (F.mapTriangle.map g)
        ⊢ Eq (F.map f.hom₂) (F.map g.hom₂)
      -/
    · exact congr_arg TriangleMorphism.hom₂ h
      /-
        🎉 no goals
      -/
      /-
        case h₃.a
        C : Type u_1
        D : Type u_2
        E : Type u_3
        inst✝⁸ : CategoryTheory.Category.{u_4, u_1} C
        inst✝⁷ : CategoryTheory.Category.{u_5, u_2} D
        inst✝⁶ : CategoryTheory.Category.{?u.9225, u_3} E
        inst✝⁵ : CategoryTheory.HasShift C Int
        inst✝⁴ : CategoryTheory.HasShift D Int
        inst✝³ : CategoryTheory.HasShift E Int
        F : CategoryTheory.Functor C D
        inst✝² : F.CommShift Int
        G : CategoryTheory.Functor D E
        inst✝¹ : G.CommShift Int
        inst✝ : F.Faithful
        X Y : CategoryTheory.Pretriangulated.Triangle C
        f g : Quiver.Hom X Y
        h : Eq (F.mapTriangle.map f) (F.mapTriangle.map g)
        ⊢ Eq (F.map f.hom₃) (F.map g.hom₃)
      -/
    · exact congr_arg TriangleMorphism.hom₃ h
      /-
        🎉 no goals
      -/


instance [Full F] [Faithful F] : Full F.mapTriangle where
  map_surjective {X Y} f :=
   ⟨{ hom₁ := F.preimage f.hom₁
      hom₂ := F.preimage f.hom₂
      hom₃ := F.preimage f.hom₃
      comm₁ := F.map_injective
            /-
              C : Type u_1
              D : Type u_2
              E : Type u_3
              inst✝⁹ : CategoryTheory.Category.{u_4, u_1} C
              inst✝⁸ : CategoryTheory.Category.{u_5, u_2} D
              inst✝⁷ : CategoryTheory.Category.{?u.11556, u_3} E
              inst✝⁶ : CategoryTheory.HasShift C Int
              inst✝⁵ : CategoryTheory.HasShift D Int
              inst✝⁴ : CategoryTheory.HasShift E Int
              F : CategoryTheory.Functor C D
              inst✝³ : F.CommShift Int
              G : CategoryTheory.Functor D E
              inst✝² : G.CommShift Int
              inst✝¹ : F.Full
              inst✝ : F.Faithful
              X Y : CategoryTheory.Pretriangulated.Triangle C
              f : Quiver.Hom (F.mapTriangle.obj X) (F.mapTriangle.obj Y)
              ⊢ Eq (F.map (CategoryTheory.CategoryStruct.comp X.mor₁ (F.preimage f.hom₂))) ( …
            -/
        (by simpa only [mapTriangle_obj, map_comp, map_preimage] using f.comm₁)
            /-
              🎉 no goals
            -/
      comm₂ := F.map_injective
            /-
              C : Type u_1
              D : Type u_2
              E : Type u_3
              inst✝⁹ : CategoryTheory.Category.{u_4, u_1} C
              inst✝⁸ : CategoryTheory.Category.{u_5, u_2} D
              inst✝⁷ : CategoryTheory.Category.{?u.11556, u_3} E
              inst✝⁶ : CategoryTheory.HasShift C Int
              inst✝⁵ : CategoryTheory.HasShift D Int
              inst✝⁴ : CategoryTheory.HasShift E Int
              F : CategoryTheory.Functor C D
              inst✝³ : F.CommShift Int
              G : CategoryTheory.Functor D E
              inst✝² : G.CommShift Int
              inst✝¹ : F.Full
              inst✝ : F.Faithful
              X Y : CategoryTheory.Pretriangulated.Triangle C
              f : Quiver.Hom (F.mapTriangle.obj X) (F.mapTriangle.obj Y)
              ⊢ Eq (F.map (CategoryTheory.CategoryStruct.comp X.mor₂ (F.preimage f.hom₃))) ( …
            -/
        (by simpa only [mapTriangle_obj, map_comp, map_preimage] using f.comm₂)
            /-
              🎉 no goals
            -/
      comm₃ := F.map_injective (by
        /-
          C : Type u_1
          D : Type u_2
          E : Type u_3
          inst✝⁹ : CategoryTheory.Category.{u_4, u_1} C
          inst✝⁸ : CategoryTheory.Category.{u_5, u_2} D
          inst✝⁷ : CategoryTheory.Category.{?u.11556, u_3} E
          inst✝⁶ : CategoryTheory.HasShift C Int
          inst✝⁵ : CategoryTheory.HasShift D Int
          inst✝⁴ : CategoryTheory.HasShift E Int
          F : CategoryTheory.Functor C D
          inst✝³ : F.CommShift Int
          G : CategoryTheory.Functor D E
          inst✝² : G.CommShift Int
          inst✝¹ : F.Full
          inst✝ : F.Faithful
          X Y : CategoryTheory.Pretriangulated.Triangle C
          f : Quiver.Hom (F.mapTriangle.obj X) (F.mapTriangle.obj Y)
          ⊢ Eq (F.map (CategoryTheory.CategoryStruct.comp X.mor₃ ((CategoryTheory.shiftF …
        -/
        rw [← cancel_mono ((F.commShiftIso (1 : ℤ)).hom.app Y.obj₁)]
        simpa only [mapTriangle_obj, map_comp, assoc, commShiftIso_hom_naturality,
                                                               /-
                                                                 C : Type u_1
                                                                 D : Type u_2
                                                                 E : Type u_3
                                                                 inst✝⁹ : CategoryTheory.Category.{u_4, u_1} C
                                                                 inst✝⁸ : CategoryTheory.Category.{u_5, u_2} D
                                                                 inst✝⁷ : CategoryTheory.Category.{?u.11556, u_3} E
                                                                 inst✝⁶ : CategoryTheory.HasShift C Int
                                                                 inst✝⁵ : CategoryTheory.HasShift D Int
                                                                 inst✝⁴ : CategoryTheory.HasShift E Int
                                                                 F : CategoryTheory.Functor C D
                                                                 inst✝³ : F.CommShift Int
                                                                 G : CategoryTheory.Functor D E
                                                                 inst✝² : G.CommShift Int
                                                                 inst✝¹ : F.Full
                                                                 inst✝ : F.Faithful
                                                                 X Y : CategoryTheory.Pretriangulated.Triangle C
                                                                 f : Quiver.Hom (F.mapTriangle.obj X) (F.mapTriangle.obj Y)
                                                                 ⊢ Eq (F.mapTriangle.map { hom₁ := F.preimage f.hom₁, hom₂ := F.preimage f.hom₂ …
                                                               -/
          map_preimage, Triangle.mk_mor₃] using f.comm₃) }, by aesop_cat⟩
                                                               /-
                                                                 🎉 no goals
                                                               -/


/-- The functor `F.mapTriangle` commutes with the shift. -/
noncomputable def mapTriangleCommShiftIso (n : ℤ) :
    Triangle.shiftFunctor C n ⋙ F.mapTriangle ≅ F.mapTriangle ⋙ Triangle.shiftFunctor D n :=
  NatIso.ofComponents (fun T => Triangle.isoMk _ _
    ((F.commShiftIso n).app _) ((F.commShiftIso n).app _) ((F.commShiftIso n).app _)
        /-
          C : Type u_1
          D : Type u_2
          E : Type u_3
          inst✝¹⁰ : CategoryTheory.Category.{?u.22257, u_1} C
          inst✝⁹ : CategoryTheory.Category.{?u.22261, u_2} D
          inst✝⁸ : CategoryTheory.Category.{?u.22265, u_3} E
          inst✝⁷ : CategoryTheory.HasShift C Int
          inst✝⁶ : CategoryTheory.HasShift D Int
          inst✝⁵ : CategoryTheory.HasShift E Int
          F : CategoryTheory.Functor C D
          inst✝⁴ : F.CommShift Int
          G : CategoryTheory.Functor D E
          inst✝³ : G.CommShift Int
          inst✝² : CategoryTheory.Preadditive C
          inst✝¹ : CategoryTheory.Preadditive D
          inst✝ : F.Additive
          n : Int
          T : CategoryTheory.Pretriangulated.Triangle C
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Pretriangulated.Tri …
        -/
        /-
          🎉 no goals
        -/
    (by aesop_cat) (by aesop_cat) (by
                       /-
                         🎉 no goals
                       -/
      /-
        C : Type u_1
        D : Type u_2
        E : Type u_3
        inst✝¹⁰ : CategoryTheory.Category.{?u.22257, u_1} C
        inst✝⁹ : CategoryTheory.Category.{?u.22261, u_2} D
        inst✝⁸ : CategoryTheory.Category.{?u.22265, u_3} E
        inst✝⁷ : CategoryTheory.HasShift C Int
        inst✝⁶ : CategoryTheory.HasShift D Int
        inst✝⁵ : CategoryTheory.HasShift E Int
        F : CategoryTheory.Functor C D
        inst✝⁴ : F.CommShift Int
        G : CategoryTheory.Functor D E
        inst✝³ : G.CommShift Int
        inst✝² : CategoryTheory.Preadditive C
        inst✝¹ : CategoryTheory.Preadditive D
        inst✝ : F.Additive
        n : Int
        T : CategoryTheory.Pretriangulated.Triangle C
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Pretriangulated.Tri …
      -/
      dsimp
      simp only [map_units_smul, map_comp, Linear.units_smul_comp, assoc,
        Linear.comp_units_smul, ← F.commShiftIso_hom_naturality_assoc]
      /-
        C : Type u_1
        D : Type u_2
        E : Type u_3
        inst✝¹⁰ : CategoryTheory.Category.{?u.22257, u_1} C
        inst✝⁹ : CategoryTheory.Category.{?u.22261, u_2} D
        inst✝⁸ : CategoryTheory.Category.{?u.22265, u_3} E
        inst✝⁷ : CategoryTheory.HasShift C Int
        inst✝⁶ : CategoryTheory.HasShift D Int
        inst✝⁵ : CategoryTheory.HasShift E Int
        F : CategoryTheory.Functor C D
        inst✝⁴ : F.CommShift Int
        G : CategoryTheory.Functor D E
        inst✝³ : G.CommShift Int
        inst✝² : CategoryTheory.Preadditive C
        inst✝¹ : CategoryTheory.Preadditive D
        inst✝ : F.Additive
        n : Int
        T : CategoryTheory.Pretriangulated.Triangle C
        ⊢ Eq (HSMul.hSMul n.negOnePow (CategoryTheory.CategoryStruct.comp (F.map ((Cat …
      -/
      rw [F.map_shiftFunctorComm_hom_app T.obj₁ 1 n]
      simp only [comp_obj, assoc, Iso.inv_hom_id_app_assoc,
                                                                       /-
                                                                         C : Type u_1
                                                                         D : Type u_2
                                                                         E : Type u_3
                                                                         inst✝¹⁰ : CategoryTheory.Category.{?u.22257, u_1} C
                                                                         inst✝⁹ : CategoryTheory.Category.{?u.22261, u_2} D
                                                                         inst✝⁸ : CategoryTheory.Category.{?u.22265, u_3} E
                                                                         inst✝⁷ : CategoryTheory.HasShift C Int
                                                                         inst✝⁶ : CategoryTheory.HasShift D Int
                                                                         inst✝⁵ : CategoryTheory.HasShift E Int
                                                                         F : CategoryTheory.Functor C D
                                                                         inst✝⁴ : F.CommShift Int
                                                                         G : CategoryTheory.Functor D E
                                                                         inst✝³ : G.CommShift Int
                                                                         inst✝² : CategoryTheory.Preadditive C
                                                                         inst✝¹ : CategoryTheory.Preadditive D
                                                                         inst✝ : F.Additive
                                                                         n : Int
                                                                         ⊢ ∀ {X Y : CategoryTheory.Pretriangulated.Triangle C} (f : Quiver.Hom X Y), Eq …
                                                                       -/
        ← Functor.map_comp, Iso.inv_hom_id_app, map_id, comp_id])) (by aesop_cat)
                                                                       /-
                                                                         🎉 no goals
                                                                       -/


attribute [simps!] mapTriangleCommShiftIso


set_option maxHeartbeats 400000 in
noncomputable instance [∀ (n : ℤ), (shiftFunctor C n).Additive]
    [∀ (n : ℤ), (shiftFunctor D n).Additive] : (F.mapTriangle).CommShift ℤ where
  iso := F.mapTriangleCommShiftIso


/-- `F.mapTriangle` commutes with the rotation of triangles. -/
@[simps!]
def mapTriangleRotateIso :
    F.mapTriangle ⋙ Pretriangulated.rotate D ≅
      Pretriangulated.rotate C ⋙ F.mapTriangle :=
  NatIso.ofComponents
    (fun T => Triangle.isoMk _ _ (Iso.refl _) (Iso.refl _)
      ((F.commShiftIso (1 : ℤ)).symm.app _)
          /-
            C : Type u_1
            D : Type u_2
            E : Type u_3
            inst✝¹⁰ : CategoryTheory.Category.{?u.194631, u_1} C
            inst✝⁹ : CategoryTheory.Category.{?u.194635, u_2} D
            inst✝⁸ : CategoryTheory.Category.{?u.194639, u_3} E
            inst✝⁷ : CategoryTheory.HasShift C Int
            inst✝⁶ : CategoryTheory.HasShift D Int
            inst✝⁵ : CategoryTheory.HasShift E Int
            F : CategoryTheory.Functor C D
            inst✝⁴ : F.CommShift Int
            G : CategoryTheory.Functor D E
            inst✝³ : G.CommShift Int
            inst✝² : CategoryTheory.Preadditive C
            inst✝¹ : CategoryTheory.Preadditive D
            inst✝ : F.Additive
            T : CategoryTheory.Pretriangulated.Triangle C
            ⊢ Eq (CategoryTheory.CategoryStruct.comp ((F.mapTriangle.comp (CategoryTheory. …
          -/
          /-
            🎉 no goals
          -/
                         /-
                           🎉 no goals
                         -/
                                        /-
                                          🎉 no goals
                                        -/
      (by aesop_cat) (by aesop_cat) (by aesop_cat)) (by aesop_cat)
                                                        /-
                                                          🎉 no goals
                                                        -/


/-- `F.mapTriangle` commutes with the inverse of the rotation of triangles. -/
@[simps!]
noncomputable def mapTriangleInvRotateIso [F.Additive] :
    F.mapTriangle ⋙ Pretriangulated.invRotate D ≅
      Pretriangulated.invRotate C ⋙ F.mapTriangle :=
  NatIso.ofComponents
    (fun T => Triangle.isoMk _ _ ((F.commShiftIso (-1 : ℤ)).symm.app _) (Iso.refl _) (Iso.refl _)
          /-
            C : Type u_1
            D : Type u_2
            E : Type u_3
            inst✝¹¹ : CategoryTheory.Category.{?u.224308, u_1} C
            inst✝¹⁰ : CategoryTheory.Category.{?u.224312, u_2} D
            inst✝⁹ : CategoryTheory.Category.{?u.224316, u_3} E
            inst✝⁸ : CategoryTheory.HasShift C Int
            inst✝⁷ : CategoryTheory.HasShift D Int
            inst✝⁶ : CategoryTheory.HasShift E Int
            F : CategoryTheory.Functor C D
            inst✝⁵ : F.CommShift Int
            G : CategoryTheory.Functor D E
            inst✝⁴ : G.CommShift Int
            inst✝³ : CategoryTheory.Preadditive C
            inst✝² : CategoryTheory.Preadditive D
            inst✝¹ inst✝ : F.Additive
            T : CategoryTheory.Pretriangulated.Triangle C
            ⊢ Eq (CategoryTheory.CategoryStruct.comp ((F.mapTriangle.comp (CategoryTheory. …
          -/
          /-
            🎉 no goals
          -/
                         /-
                           🎉 no goals
                         -/
                                        /-
                                          🎉 no goals
                                        -/
      (by aesop_cat) (by aesop_cat) (by aesop_cat)) (by aesop_cat)
                                                        /-
                                                          🎉 no goals
                                                        -/



variable (C) in
/-- The canonical isomorphism `(𝟭 C).mapTriangle ≅ 𝟭 (Triangle C)`. -/
@[simps!]
def mapTriangleIdIso : (𝟭 C).mapTriangle ≅ 𝟭 _ :=
                               /-
                                 C : Type u_1
                                 D : Type u_2
                                 E : Type u_3
                                 inst✝¹⁰ : CategoryTheory.Category.{?u.261069, u_1} C
                                 inst✝⁹ : CategoryTheory.Category.{?u.261073, u_2} D
                                 inst✝⁸ : CategoryTheory.Category.{?u.261077, u_3} E
                                 inst✝⁷ : CategoryTheory.HasShift C Int
                                 inst✝⁶ : CategoryTheory.HasShift D Int
                                 inst✝⁵ : CategoryTheory.HasShift E Int
                                 F : CategoryTheory.Functor C D
                                 inst✝⁴ : F.CommShift Int
                                 G : CategoryTheory.Functor D E
                                 inst✝³ : G.CommShift Int
                                 inst✝² : CategoryTheory.Preadditive C
                                 inst✝¹ : CategoryTheory.Preadditive D
                                 inst✝ : F.Additive
                                 T : CategoryTheory.Pretriangulated.Triangle C
                                 ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Functor.id C).mapTri …
                               -/
                               /-
                                 🎉 no goals
                               -/
                               /-
                                 🎉 no goals
                               -/
                               /-
                                 🎉 no goals
                               -/
  NatIso.ofComponents (fun T ↦ Triangle.isoMk _ _ (Iso.refl _) (Iso.refl _) (Iso.refl _))
  /-
    🎉 no goals
  -/


/-- The canonical isomorphism `(F ⋙ G).mapTriangle ≅ F.mapTriangle ⋙ G.mapTriangle`. -/
@[simps!]
def mapTriangleCompIso : (F ⋙ G).mapTriangle ≅ F.mapTriangle ⋙ G.mapTriangle :=
                                /-
                                  C : Type u_1
                                  D : Type u_2
                                  E : Type u_3
                                  inst✝¹⁰ : CategoryTheory.Category.{?u.279345, u_1} C
                                  inst✝⁹ : CategoryTheory.Category.{?u.279349, u_2} D
                                  inst✝⁸ : CategoryTheory.Category.{?u.279353, u_3} E
                                  inst✝⁷ : CategoryTheory.HasShift C Int
                                  inst✝⁶ : CategoryTheory.HasShift D Int
                                  inst✝⁵ : CategoryTheory.HasShift E Int
                                  F : CategoryTheory.Functor C D
                                  inst✝⁴ : F.CommShift Int
                                  G : CategoryTheory.Functor D E
                                  inst✝³ : G.CommShift Int
                                  inst✝² : CategoryTheory.Preadditive C
                                  inst✝¹ : CategoryTheory.Preadditive D
                                  inst✝ : F.Additive
                                  T : CategoryTheory.Pretriangulated.Triangle C
                                  ⊢ Eq (CategoryTheory.CategoryStruct.comp ((F.comp G).mapTriangle.obj T).mor₁ ( …
                                -/
                                /-
                                  🎉 no goals
                                -/
                                /-
                                  🎉 no goals
                                -/
                                /-
                                  🎉 no goals
                                -/
  NatIso.ofComponents (fun T => Triangle.isoMk _ _ (Iso.refl _) (Iso.refl _) (Iso.refl _))
  /-
    🎉 no goals
  -/


/-- Two isomorphic functors `F₁` and `F₂` induce isomorphic functors
`F₁.mapTriangle` and `F₂.mapTriangle` if the isomorphism `F₁ ≅ F₂` is compatible
with the shifts. -/
@[simps!]
def mapTriangleIso {F₁ F₂ : C ⥤ D} (e : F₁ ≅ F₂) [F₁.CommShift ℤ] [F₂.CommShift ℤ]
    [NatTrans.CommShift e.hom ℤ] : F₁.mapTriangle ≅ F₂.mapTriangle :=
  NatIso.ofComponents (fun T =>
                                                         /-
                                                           C : Type u_1
                                                           D : Type u_2
                                                           E : Type u_3
                                                           inst✝¹³ : CategoryTheory.Category.{?u.321828, u_1} C
                                                           inst✝¹² : CategoryTheory.Category.{?u.321832, u_2} D
                                                           inst✝¹¹ : CategoryTheory.Category.{?u.321836, u_3} E
                                                           inst✝¹⁰ : CategoryTheory.HasShift C Int
                                                           inst✝⁹ : CategoryTheory.HasShift D Int
                                                           inst✝⁸ : CategoryTheory.HasShift E Int
                                                           F : CategoryTheory.Functor C D
                                                           inst✝⁷ : F.CommShift Int
                                                           G : CategoryTheory.Functor D E
                                                           inst✝⁶ : G.CommShift Int
                                                           inst✝⁵ : CategoryTheory.Preadditive C
                                                           inst✝⁴ : CategoryTheory.Preadditive D
                                                           inst✝³ : F.Additive
                                                           F₁ F₂ : CategoryTheory.Functor C D
                                                           e : CategoryTheory.Iso F₁ F₂
                                                           inst✝² : F₁.CommShift Int
                                                           inst✝¹ : F₂.CommShift Int
                                                           inst✝ : CategoryTheory.NatTrans.CommShift e.hom Int
                                                           T : CategoryTheory.Pretriangulated.Triangle C
                                                           ⊢ Eq (CategoryTheory.CategoryStruct.comp (F₁.mapTriangle.obj T).mor₁ (e.app T. …
                                                         -/
                                                         /-
                                                           🎉 no goals
                                                         -/
    Triangle.isoMk _ _ (e.app _) (e.app _) (e.app _) (by simp) (by simp) (by
                                                                   /-
                                                                     🎉 no goals
                                                                   -/
      /-
        C : Type u_1
        D : Type u_2
        E : Type u_3
        inst✝¹³ : CategoryTheory.Category.{?u.321828, u_1} C
        inst✝¹² : CategoryTheory.Category.{?u.321832, u_2} D
        inst✝¹¹ : CategoryTheory.Category.{?u.321836, u_3} E
        inst✝¹⁰ : CategoryTheory.HasShift C Int
        inst✝⁹ : CategoryTheory.HasShift D Int
        inst✝⁸ : CategoryTheory.HasShift E Int
        F : CategoryTheory.Functor C D
        inst✝⁷ : F.CommShift Int
        G : CategoryTheory.Functor D E
        inst✝⁶ : G.CommShift Int
        inst✝⁵ : CategoryTheory.Preadditive C
        inst✝⁴ : CategoryTheory.Preadditive D
        inst✝³ : F.Additive
        F₁ F₂ : CategoryTheory.Functor C D
        e : CategoryTheory.Iso F₁ F₂
        inst✝² : F₁.CommShift Int
        inst✝¹ : F₂.CommShift Int
        inst✝ : CategoryTheory.NatTrans.CommShift e.hom Int
        T : CategoryTheory.Pretriangulated.Triangle C
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (F₁.mapTriangle.obj T).mor₃ ((Categor …
      -/
      dsimp
      simp only [assoc, NatTrans.shift_app_comm e.hom (1 : ℤ) T.obj₁,
                                         /-
                                           C : Type u_1
                                           D : Type u_2
                                           E : Type u_3
                                           inst✝¹³ : CategoryTheory.Category.{?u.321828, u_1} C
                                           inst✝¹² : CategoryTheory.Category.{?u.321832, u_2} D
                                           inst✝¹¹ : CategoryTheory.Category.{?u.321836, u_3} E
                                           inst✝¹⁰ : CategoryTheory.HasShift C Int
                                           inst✝⁹ : CategoryTheory.HasShift D Int
                                           inst✝⁸ : CategoryTheory.HasShift E Int
                                           F : CategoryTheory.Functor C D
                                           inst✝⁷ : F.CommShift Int
                                           G : CategoryTheory.Functor D E
                                           inst✝⁶ : G.CommShift Int
                                           inst✝⁵ : CategoryTheory.Preadditive C
                                           inst✝⁴ : CategoryTheory.Preadditive D
                                           inst✝³ : F.Additive
                                           F₁ F₂ : CategoryTheory.Functor C D
                                           e : CategoryTheory.Iso F₁ F₂
                                           inst✝² : F₁.CommShift Int
                                           inst✝¹ : F₂.CommShift Int
                                           inst✝ : CategoryTheory.NatTrans.CommShift e.hom Int
                                           ⊢ ∀ {X Y : CategoryTheory.Pretriangulated.Triangle C} (f : Quiver.Hom X Y), Eq …
                                         -/
        NatTrans.naturality_assoc])) (by aesop_cat)
                                         /-
                                           🎉 no goals
                                         -/


/-- A functor which commutes with the shift by `ℤ` is triangulated if
it sends distinguished triangles to distinguished triangles. -/
class IsTriangulated : Prop where
  map_distinguished (T : Triangle C) : (T ∈ distTriang C) → F.mapTriangle.obj T ∈ distTriang D


lemma map_distinguished [F.IsTriangulated] (T : Triangle C) (hT : T ∈ distTriang C) :
    F.mapTriangle.obj T ∈ distTriang D :=
  IsTriangulated.map_distinguished _ hT


instance (priority := 100) [F.IsTriangulated] : PreservesZeroMorphisms F where
  map_zero X Y := by
    /-
      C : Type u_1
      D : Type u_2
      E : Type u_3
      inst✝²⁰ : CategoryTheory.Category.{u_4, u_1} C
      inst✝¹⁹ : CategoryTheory.Category.{u_5, u_2} D
      inst✝¹⁸ : CategoryTheory.Category.{?u.339080, u_3} E
      inst✝¹⁷ : CategoryTheory.HasShift C Int
      inst✝¹⁶ : CategoryTheory.HasShift D Int
      inst✝¹⁵ : CategoryTheory.HasShift E Int
      F : CategoryTheory.Functor C D
      inst✝¹⁴ : F.CommShift Int
      G : CategoryTheory.Functor D E
      inst✝¹³ : G.CommShift Int
      inst✝¹² : CategoryTheory.Limits.HasZeroObject C
      inst✝¹¹ : CategoryTheory.Limits.HasZeroObject D
      inst✝¹⁰ : CategoryTheory.Limits.HasZeroObject E
      inst✝⁹ : CategoryTheory.Preadditive C
      inst✝⁸ : CategoryTheory.Preadditive D
      inst✝⁷ : CategoryTheory.Preadditive E
      inst✝⁶ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      inst✝⁵ : ∀ (n : Int), (CategoryTheory.shiftFunctor D n).Additive
      inst✝⁴ : ∀ (n : Int), (CategoryTheory.shiftFunctor E n).Additive
      inst✝³ : CategoryTheory.Pretriangulated C
      inst✝² : CategoryTheory.Pretriangulated D
      inst✝¹ : CategoryTheory.Pretriangulated E
      inst✝ : F.IsTriangulated
      X Y : C
      ⊢ Eq (F.map 0) 0
    -/
    have h₁ : (0 : X ⟶ Y) = 0 ≫ 𝟙 0 ≫ 0 := by simp
    have h₂ : 𝟙 (F.obj 0) = 0 := by
      rw [← IsZero.iff_id_eq_zero]
      apply Triangle.isZero₃_of_isIso₁ _
        (F.map_distinguished _ (contractible_distinguished (0 : C)))
      dsimp
      infer_instance
    /-
      C : Type u_1
      D : Type u_2
      E : Type u_3
      inst✝²⁰ : CategoryTheory.Category.{u_4, u_1} C
      inst✝¹⁹ : CategoryTheory.Category.{u_5, u_2} D
      inst✝¹⁸ : CategoryTheory.Category.{?u.339080, u_3} E
      inst✝¹⁷ : CategoryTheory.HasShift C Int
      inst✝¹⁶ : CategoryTheory.HasShift D Int
      inst✝¹⁵ : CategoryTheory.HasShift E Int
      F : CategoryTheory.Functor C D
      inst✝¹⁴ : F.CommShift Int
      G : CategoryTheory.Functor D E
      inst✝¹³ : G.CommShift Int
      inst✝¹² : CategoryTheory.Limits.HasZeroObject C
      inst✝¹¹ : CategoryTheory.Limits.HasZeroObject D
      inst✝¹⁰ : CategoryTheory.Limits.HasZeroObject E
      inst✝⁹ : CategoryTheory.Preadditive C
      inst✝⁸ : CategoryTheory.Preadditive D
      inst✝⁷ : CategoryTheory.Preadditive E
      inst✝⁶ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      inst✝⁵ : ∀ (n : Int), (CategoryTheory.shiftFunctor D n).Additive
      inst✝⁴ : ∀ (n : Int), (CategoryTheory.shiftFunctor E n).Additive
      inst✝³ : CategoryTheory.Pretriangulated C
      inst✝² : CategoryTheory.Pretriangulated D
      inst✝¹ : CategoryTheory.Pretriangulated E
      inst✝ : F.IsTriangulated
      X Y : C
      h₁ : Eq 0 (CategoryTheory.CategoryStruct.comp 0 (CategoryTheory.CategoryStruct …
      h₂ : Eq (CategoryTheory.CategoryStruct.id (F.obj 0)) 0
      ⊢ Eq (F.map 0) 0
    -/
    rw [h₁, F.map_comp, F.map_comp, F.map_id, h₂, zero_comp, comp_zero]
    /-
      🎉 no goals
    -/


noncomputable instance [F.IsTriangulated] :
    PreservesLimitsOfShape (Discrete WalkingPair) F := by
  suffices ∀ (X₁ X₃ : C), IsIso (prodComparison F X₁ X₃) by
    have := fun (X₁ X₃ : C) ↦ PreservesLimitPair.of_iso_prod_comparison F X₁ X₃
    exact ⟨fun {K} ↦ preservesLimit_of_iso_diagram F (diagramIsoPair K).symm⟩
  /-
    C : Type u_1
    D : Type u_2
    E : Type u_3
    inst✝²⁰ : CategoryTheory.Category.{u_4, u_1} C
    inst✝¹⁹ : CategoryTheory.Category.{u_5, u_2} D
    inst✝¹⁸ : CategoryTheory.Category.{?u.343608, u_3} E
    inst✝¹⁷ : CategoryTheory.HasShift C Int
    inst✝¹⁶ : CategoryTheory.HasShift D Int
    inst✝¹⁵ : CategoryTheory.HasShift E Int
    F : CategoryTheory.Functor C D
    inst✝¹⁴ : F.CommShift Int
    G : CategoryTheory.Functor D E
    inst✝¹³ : G.CommShift Int
    inst✝¹² : CategoryTheory.Limits.HasZeroObject C
    inst✝¹¹ : CategoryTheory.Limits.HasZeroObject D
    inst✝¹⁰ : CategoryTheory.Limits.HasZeroObject E
    inst✝⁹ : CategoryTheory.Preadditive C
    inst✝⁸ : CategoryTheory.Preadditive D
    inst✝⁷ : CategoryTheory.Preadditive E
    inst✝⁶ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝⁵ : ∀ (n : Int), (CategoryTheory.shiftFunctor D n).Additive
    inst✝⁴ : ∀ (n : Int), (CategoryTheory.shiftFunctor E n).Additive
    inst✝³ : CategoryTheory.Pretriangulated C
    inst✝² : CategoryTheory.Pretriangulated D
    inst✝¹ : CategoryTheory.Pretriangulated E
    inst✝ : F.IsTriangulated
    ⊢ ∀ (X₁ X₃ : C), CategoryTheory.IsIso (CategoryTheory.Limits.prodComparison F  …
  -/
  intro X₁ X₃
  let φ : F.mapTriangle.obj (binaryProductTriangle X₁ X₃) ⟶
      binaryProductTriangle (F.obj X₁) (F.obj X₃) :=
    { hom₁ := 𝟙 _
      hom₂ := prodComparison F X₁ X₃
      hom₃ := 𝟙 _
      comm₁ := by
        dsimp
        ext
        · simp only [assoc, prodComparison_fst, prod.comp_lift, comp_id, comp_zero,
            limit.lift_π, BinaryFan.mk_pt, BinaryFan.π_app_left, BinaryFan.mk_fst,
            ← F.map_comp, F.map_id]
        · simp only [assoc, prodComparison_snd, prod.comp_lift, comp_id, comp_zero,
            limit.lift_π, BinaryFan.mk_pt, BinaryFan.π_app_right, BinaryFan.mk_snd,
            ← F.map_comp, F.map_zero]
      comm₂ := by simp
      comm₃ := by simp }
  exact isIso₂_of_isIso₁₃ φ (F.map_distinguished _ (binaryProductTriangle_distinguished X₁ X₃))
    (binaryProductTriangle_distinguished _ _)
    (by dsimp [φ]; infer_instance) (by dsimp [φ]; infer_instance)


instance (priority := 100) [F.IsTriangulated] : F.Additive :=
  F.additive_of_preserves_binary_products


instance : (𝟭 C).IsTriangulated where
  map_distinguished T hT :=
    isomorphic_distinguished _ hT _ ((mapTriangleIdIso C).app T)


instance [F.IsTriangulated] [G.IsTriangulated] : (F ⋙ G).IsTriangulated where
  map_distinguished T hT :=
    isomorphic_distinguished _ (G.map_distinguished _ (F.map_distinguished T hT)) _
      ((mapTriangleCompIso F G).app T)


lemma isTriangulated_of_iso {F₁ F₂ : C ⥤ D} (e : F₁ ≅ F₂) [F₁.CommShift ℤ] [F₂.CommShift ℤ]
    [NatTrans.CommShift e.hom ℤ] [F₁.IsTriangulated] : F₂.IsTriangulated where
  map_distinguished T hT :=
    isomorphic_distinguished _ (F₁.map_distinguished T hT) _ ((mapTriangleIso e).app T).symm


lemma isTriangulated_iff_of_iso {F₁ F₂ : C ⥤ D} (e : F₁ ≅ F₂) [F₁.CommShift ℤ] [F₂.CommShift ℤ]
    [NatTrans.CommShift e.hom ℤ] : F₁.IsTriangulated ↔ F₂.IsTriangulated := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝¹⁴ : CategoryTheory.Category.{u_4, u_1} C
    inst✝¹³ : CategoryTheory.Category.{u_5, u_2} D
    inst✝¹² : CategoryTheory.HasShift C Int
    inst✝¹¹ : CategoryTheory.HasShift D Int
    inst✝¹⁰ : CategoryTheory.Limits.HasZeroObject C
    inst✝⁹ : CategoryTheory.Limits.HasZeroObject D
    inst✝⁸ : CategoryTheory.Preadditive C
    inst✝⁷ : CategoryTheory.Preadditive D
    inst✝⁶ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝⁵ : ∀ (n : Int), (CategoryTheory.shiftFunctor D n).Additive
    inst✝⁴ : CategoryTheory.Pretriangulated C
    inst✝³ : CategoryTheory.Pretriangulated D
    F₁ F₂ : CategoryTheory.Functor C D
    e : CategoryTheory.Iso F₁ F₂
    inst✝² : F₁.CommShift Int
    inst✝¹ : F₂.CommShift Int
    inst✝ : CategoryTheory.NatTrans.CommShift e.hom Int
    ⊢ Iff F₁.IsTriangulated F₂.IsTriangulated
  -/
  constructor
    /-
      case mp
      C : Type u_1
      D : Type u_2
      inst✝¹⁴ : CategoryTheory.Category.{u_4, u_1} C
      inst✝¹³ : CategoryTheory.Category.{u_5, u_2} D
      inst✝¹² : CategoryTheory.HasShift C Int
      inst✝¹¹ : CategoryTheory.HasShift D Int
      inst✝¹⁰ : CategoryTheory.Limits.HasZeroObject C
      inst✝⁹ : CategoryTheory.Limits.HasZeroObject D
      inst✝⁸ : CategoryTheory.Preadditive C
      inst✝⁷ : CategoryTheory.Preadditive D
      inst✝⁶ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      inst✝⁵ : ∀ (n : Int), (CategoryTheory.shiftFunctor D n).Additive
      inst✝⁴ : CategoryTheory.Pretriangulated C
      inst✝³ : CategoryTheory.Pretriangulated D
      F₁ F₂ : CategoryTheory.Functor C D
      e : CategoryTheory.Iso F₁ F₂
      inst✝² : F₁.CommShift Int
      inst✝¹ : F₂.CommShift Int
      inst✝ : CategoryTheory.NatTrans.CommShift e.hom Int
      ⊢ F₁.IsTriangulated → F₂.IsTriangulated
    -/
  · intro
    /-
      case mp
      C : Type u_1
      D : Type u_2
      inst✝¹⁴ : CategoryTheory.Category.{u_4, u_1} C
      inst✝¹³ : CategoryTheory.Category.{u_5, u_2} D
      inst✝¹² : CategoryTheory.HasShift C Int
      inst✝¹¹ : CategoryTheory.HasShift D Int
      inst✝¹⁰ : CategoryTheory.Limits.HasZeroObject C
      inst✝⁹ : CategoryTheory.Limits.HasZeroObject D
      inst✝⁸ : CategoryTheory.Preadditive C
      inst✝⁷ : CategoryTheory.Preadditive D
      inst✝⁶ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      inst✝⁵ : ∀ (n : Int), (CategoryTheory.shiftFunctor D n).Additive
      inst✝⁴ : CategoryTheory.Pretriangulated C
      inst✝³ : CategoryTheory.Pretriangulated D
      F₁ F₂ : CategoryTheory.Functor C D
      e : CategoryTheory.Iso F₁ F₂
      inst✝² : F₁.CommShift Int
      inst✝¹ : F₂.CommShift Int
      inst✝ : CategoryTheory.NatTrans.CommShift e.hom Int
      a✝ : F₁.IsTriangulated
      ⊢ F₂.IsTriangulated
    -/
    exact isTriangulated_of_iso e
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u_1
      D : Type u_2
      inst✝¹⁴ : CategoryTheory.Category.{u_4, u_1} C
      inst✝¹³ : CategoryTheory.Category.{u_5, u_2} D
      inst✝¹² : CategoryTheory.HasShift C Int
      inst✝¹¹ : CategoryTheory.HasShift D Int
      inst✝¹⁰ : CategoryTheory.Limits.HasZeroObject C
      inst✝⁹ : CategoryTheory.Limits.HasZeroObject D
      inst✝⁸ : CategoryTheory.Preadditive C
      inst✝⁷ : CategoryTheory.Preadditive D
      inst✝⁶ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      inst✝⁵ : ∀ (n : Int), (CategoryTheory.shiftFunctor D n).Additive
      inst✝⁴ : CategoryTheory.Pretriangulated C
      inst✝³ : CategoryTheory.Pretriangulated D
      F₁ F₂ : CategoryTheory.Functor C D
      e : CategoryTheory.Iso F₁ F₂
      inst✝² : F₁.CommShift Int
      inst✝¹ : F₂.CommShift Int
      inst✝ : CategoryTheory.NatTrans.CommShift e.hom Int
      ⊢ F₂.IsTriangulated → F₁.IsTriangulated
    -/
  · intro
    /-
      case mpr
      C : Type u_1
      D : Type u_2
      inst✝¹⁴ : CategoryTheory.Category.{u_4, u_1} C
      inst✝¹³ : CategoryTheory.Category.{u_5, u_2} D
      inst✝¹² : CategoryTheory.HasShift C Int
      inst✝¹¹ : CategoryTheory.HasShift D Int
      inst✝¹⁰ : CategoryTheory.Limits.HasZeroObject C
      inst✝⁹ : CategoryTheory.Limits.HasZeroObject D
      inst✝⁸ : CategoryTheory.Preadditive C
      inst✝⁷ : CategoryTheory.Preadditive D
      inst✝⁶ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      inst✝⁵ : ∀ (n : Int), (CategoryTheory.shiftFunctor D n).Additive
      inst✝⁴ : CategoryTheory.Pretriangulated C
      inst✝³ : CategoryTheory.Pretriangulated D
      F₁ F₂ : CategoryTheory.Functor C D
      e : CategoryTheory.Iso F₁ F₂
      inst✝² : F₁.CommShift Int
      inst✝¹ : F₂.CommShift Int
      inst✝ : CategoryTheory.NatTrans.CommShift e.hom Int
      a✝ : F₂.IsTriangulated
      ⊢ F₁.IsTriangulated
    -/
    have : NatTrans.CommShift e.symm.hom ℤ := inferInstanceAs (NatTrans.CommShift e.inv ℤ)
    /-
      case mpr
      C : Type u_1
      D : Type u_2
      inst✝¹⁴ : CategoryTheory.Category.{u_4, u_1} C
      inst✝¹³ : CategoryTheory.Category.{u_5, u_2} D
      inst✝¹² : CategoryTheory.HasShift C Int
      inst✝¹¹ : CategoryTheory.HasShift D Int
      inst✝¹⁰ : CategoryTheory.Limits.HasZeroObject C
      inst✝⁹ : CategoryTheory.Limits.HasZeroObject D
      inst✝⁸ : CategoryTheory.Preadditive C
      inst✝⁷ : CategoryTheory.Preadditive D
      inst✝⁶ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      inst✝⁵ : ∀ (n : Int), (CategoryTheory.shiftFunctor D n).Additive
      inst✝⁴ : CategoryTheory.Pretriangulated C
      inst✝³ : CategoryTheory.Pretriangulated D
      F₁ F₂ : CategoryTheory.Functor C D
      e : CategoryTheory.Iso F₁ F₂
      inst✝² : F₁.CommShift Int
      inst✝¹ : F₂.CommShift Int
      inst✝ : CategoryTheory.NatTrans.CommShift e.hom Int
      a✝ : F₂.IsTriangulated
      this : CategoryTheory.NatTrans.CommShift e.symm.hom Int
      ⊢ F₁.IsTriangulated
    -/
    exact isTriangulated_of_iso e.symm
    /-
      🎉 no goals
    -/


lemma mem_mapTriangle_essImage_of_distinguished
    [F.IsTriangulated] [F.mapArrow.EssSurj] (T : Triangle D) (hT : T ∈ distTriang D) :
    ∃ (T' : Triangle C) (_ : T' ∈ distTriang C), Nonempty (F.mapTriangle.obj T' ≅ T) := by
  obtain ⟨X, Y, f, e₁, e₂, w⟩ : ∃ (X Y : C) (f : X ⟶ Y) (e₁ : F.obj X ≅ T.obj₁)
    (e₂ : F.obj Y ≅ T.obj₂), F.map f ≫ e₂.hom = e₁.hom ≫ T.mor₁ := by
      let e := F.mapArrow.objObjPreimageIso (Arrow.mk T.mor₁)
      exact ⟨_, _, _, Arrow.leftFunc.mapIso e, Arrow.rightFunc.mapIso e, e.hom.w.symm⟩
  /-
    case intro.intro.intro.intro.intro
    C : Type u_1
    D : Type u_2
    inst✝¹⁴ : CategoryTheory.Category.{u_4, u_1} C
    inst✝¹³ : CategoryTheory.Category.{u_5, u_2} D
    inst✝¹² : CategoryTheory.HasShift C Int
    inst✝¹¹ : CategoryTheory.HasShift D Int
    F : CategoryTheory.Functor C D
    inst✝¹⁰ : F.CommShift Int
    inst✝⁹ : CategoryTheory.Limits.HasZeroObject C
    inst✝⁸ : CategoryTheory.Limits.HasZeroObject D
    inst✝⁷ : CategoryTheory.Preadditive C
    inst✝⁶ : CategoryTheory.Preadditive D
    inst✝⁵ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝⁴ : ∀ (n : Int), (CategoryTheory.shiftFunctor D n).Additive
    inst✝³ : CategoryTheory.Pretriangulated C
    inst✝² : CategoryTheory.Pretriangulated D
    inst✝¹ : F.IsTriangulated
    inst✝ : F.mapArrow.EssSurj
    T : CategoryTheory.Pretriangulated.Triangle D
    hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T
    X Y : C
    f : Quiver.Hom X Y
    e₁ : CategoryTheory.Iso (F.obj X) T.obj₁
    e₂ : CategoryTheory.Iso (F.obj Y) T.obj₂
    w : Eq (CategoryTheory.CategoryStruct.comp (F.map f) e₂.hom) (CategoryTheory.C …
    ⊢ Exists fun T' => Exists fun x => Nonempty (CategoryTheory.Iso (F.mapTriangle …
  -/
  obtain ⟨W, g, h, H⟩ := distinguished_cocone_triangle f
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro
    C : Type u_1
    D : Type u_2
    inst✝¹⁴ : CategoryTheory.Category.{u_4, u_1} C
    inst✝¹³ : CategoryTheory.Category.{u_5, u_2} D
    inst✝¹² : CategoryTheory.HasShift C Int
    inst✝¹¹ : CategoryTheory.HasShift D Int
    F : CategoryTheory.Functor C D
    inst✝¹⁰ : F.CommShift Int
    inst✝⁹ : CategoryTheory.Limits.HasZeroObject C
    inst✝⁸ : CategoryTheory.Limits.HasZeroObject D
    inst✝⁷ : CategoryTheory.Preadditive C
    inst✝⁶ : CategoryTheory.Preadditive D
    inst✝⁵ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝⁴ : ∀ (n : Int), (CategoryTheory.shiftFunctor D n).Additive
    inst✝³ : CategoryTheory.Pretriangulated C
    inst✝² : CategoryTheory.Pretriangulated D
    inst✝¹ : F.IsTriangulated
    inst✝ : F.mapArrow.EssSurj
    T : CategoryTheory.Pretriangulated.Triangle D
    hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T
    X Y : C
    f : Quiver.Hom X Y
    e₁ : CategoryTheory.Iso (F.obj X) T.obj₁
    e₂ : CategoryTheory.Iso (F.obj Y) T.obj₂
    w : Eq (CategoryTheory.CategoryStruct.comp (F.map f) e₂.hom) (CategoryTheory.C …
    W : C
    g : Quiver.Hom Y W
    h : Quiver.Hom W ((CategoryTheory.shiftFunctor C 1).obj X)
    H : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Cate …
    ⊢ Exists fun T' => Exists fun x => Nonempty (CategoryTheory.Iso (F.mapTriangle …
  -/
  exact ⟨_, H, ⟨isoTriangleOfIso₁₂ _ _ (F.map_distinguished _ H) hT e₁ e₂ w⟩⟩
  /-
    🎉 no goals
  -/


lemma isTriangulated_of_precomp
    [(F ⋙ G).IsTriangulated] [F.IsTriangulated] [F.mapArrow.EssSurj] :
    G.IsTriangulated where
  map_distinguished T hT := by
    /-
      C : Type u_1
      D : Type u_2
      E : Type u_3
      inst✝²² : CategoryTheory.Category.{u_4, u_1} C
      inst✝²¹ : CategoryTheory.Category.{u_6, u_2} D
      inst✝²⁰ : CategoryTheory.Category.{u_5, u_3} E
      inst✝¹⁹ : CategoryTheory.HasShift C Int
      inst✝¹⁸ : CategoryTheory.HasShift D Int
      inst✝¹⁷ : CategoryTheory.HasShift E Int
      F : CategoryTheory.Functor C D
      inst✝¹⁶ : F.CommShift Int
      G : CategoryTheory.Functor D E
      inst✝¹⁵ : G.CommShift Int
      inst✝¹⁴ : CategoryTheory.Limits.HasZeroObject C
      inst✝¹³ : CategoryTheory.Limits.HasZeroObject D
      inst✝¹² : CategoryTheory.Limits.HasZeroObject E
      inst✝¹¹ : CategoryTheory.Preadditive C
      inst✝¹⁰ : CategoryTheory.Preadditive D
      inst✝⁹ : CategoryTheory.Preadditive E
      inst✝⁸ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      inst✝⁷ : ∀ (n : Int), (CategoryTheory.shiftFunctor D n).Additive
      inst✝⁶ : ∀ (n : Int), (CategoryTheory.shiftFunctor E n).Additive
      inst✝⁵ : CategoryTheory.Pretriangulated C
      inst✝⁴ : CategoryTheory.Pretriangulated D
      inst✝³ : CategoryTheory.Pretriangulated E
      inst✝² : (F.comp G).IsTriangulated
      inst✝¹ : F.IsTriangulated
      inst✝ : F.mapArrow.EssSurj
      T : CategoryTheory.Pretriangulated.Triangle D
      hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T
      ⊢ Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (G.mapT …
    -/
    obtain ⟨T', hT', ⟨e⟩⟩ := F.mem_mapTriangle_essImage_of_distinguished T hT
    exact isomorphic_distinguished _ ((F ⋙ G).map_distinguished T' hT') _
      (G.mapTriangle.mapIso e.symm ≪≫ (mapTriangleCompIso F G).symm.app _)


variable {F G} in
lemma isTriangulated_of_precomp_iso {H : C ⥤ E} (e : F ⋙ G ≅ H) [H.CommShift ℤ]
    [H.IsTriangulated] [F.IsTriangulated] [F.mapArrow.EssSurj] [NatTrans.CommShift e.hom ℤ] :
    G.IsTriangulated := by
  /-
    C : Type u_1
    D : Type u_2
    E : Type u_3
    inst✝²⁴ : CategoryTheory.Category.{u_4, u_1} C
    inst✝²³ : CategoryTheory.Category.{u_6, u_2} D
    inst✝²² : CategoryTheory.Category.{u_5, u_3} E
    inst✝²¹ : CategoryTheory.HasShift C Int
    inst✝²⁰ : CategoryTheory.HasShift D Int
    inst✝¹⁹ : CategoryTheory.HasShift E Int
    F : CategoryTheory.Functor C D
    inst✝¹⁸ : F.CommShift Int
    G : CategoryTheory.Functor D E
    inst✝¹⁷ : G.CommShift Int
    inst✝¹⁶ : CategoryTheory.Limits.HasZeroObject C
    inst✝¹⁵ : CategoryTheory.Limits.HasZeroObject D
    inst✝¹⁴ : CategoryTheory.Limits.HasZeroObject E
    inst✝¹³ : CategoryTheory.Preadditive C
    inst✝¹² : CategoryTheory.Preadditive D
    inst✝¹¹ : CategoryTheory.Preadditive E
    inst✝¹⁰ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝⁹ : ∀ (n : Int), (CategoryTheory.shiftFunctor D n).Additive
    inst✝⁸ : ∀ (n : Int), (CategoryTheory.shiftFunctor E n).Additive
    inst✝⁷ : CategoryTheory.Pretriangulated C
    inst✝⁶ : CategoryTheory.Pretriangulated D
    inst✝⁵ : CategoryTheory.Pretriangulated E
    H : CategoryTheory.Functor C E
    e : CategoryTheory.Iso (F.comp G) H
    inst✝⁴ : H.CommShift Int
    inst✝³ : H.IsTriangulated
    inst✝² : F.IsTriangulated
    inst✝¹ : F.mapArrow.EssSurj
    inst✝ : CategoryTheory.NatTrans.CommShift e.hom Int
    ⊢ G.IsTriangulated
  -/
  have := (isTriangulated_iff_of_iso e).2 inferInstance
  /-
    C : Type u_1
    D : Type u_2
    E : Type u_3
    inst✝²⁴ : CategoryTheory.Category.{u_4, u_1} C
    inst✝²³ : CategoryTheory.Category.{u_6, u_2} D
    inst✝²² : CategoryTheory.Category.{u_5, u_3} E
    inst✝²¹ : CategoryTheory.HasShift C Int
    inst✝²⁰ : CategoryTheory.HasShift D Int
    inst✝¹⁹ : CategoryTheory.HasShift E Int
    F : CategoryTheory.Functor C D
    inst✝¹⁸ : F.CommShift Int
    G : CategoryTheory.Functor D E
    inst✝¹⁷ : G.CommShift Int
    inst✝¹⁶ : CategoryTheory.Limits.HasZeroObject C
    inst✝¹⁵ : CategoryTheory.Limits.HasZeroObject D
    inst✝¹⁴ : CategoryTheory.Limits.HasZeroObject E
    inst✝¹³ : CategoryTheory.Preadditive C
    inst✝¹² : CategoryTheory.Preadditive D
    inst✝¹¹ : CategoryTheory.Preadditive E
    inst✝¹⁰ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝⁹ : ∀ (n : Int), (CategoryTheory.shiftFunctor D n).Additive
    inst✝⁸ : ∀ (n : Int), (CategoryTheory.shiftFunctor E n).Additive
    inst✝⁷ : CategoryTheory.Pretriangulated C
    inst✝⁶ : CategoryTheory.Pretriangulated D
    inst✝⁵ : CategoryTheory.Pretriangulated E
    H : CategoryTheory.Functor C E
    e : CategoryTheory.Iso (F.comp G) H
    inst✝⁴ : H.CommShift Int
    inst✝³ : H.IsTriangulated
    inst✝² : F.IsTriangulated
    inst✝¹ : F.mapArrow.EssSurj
    inst✝ : CategoryTheory.NatTrans.CommShift e.hom Int
    this : (F.comp G).IsTriangulated
    ⊢ G.IsTriangulated
  -/
  exact isTriangulated_of_precomp F G
  /-
    🎉 no goals
  -/


/-- The image of an octahedron by a triangulated functor. -/
@[simps]
                         /-
                           C : Type u_1
                           D : Type u_2
                           inst✝¹³ : CategoryTheory.Category.{?u.380803, u_1} C
                           inst✝¹² : CategoryTheory.Category.{?u.380807, u_2} D
                           inst✝¹¹ : CategoryTheory.HasShift C Int
                           inst✝¹⁰ : CategoryTheory.HasShift D Int
                           inst✝⁹ : CategoryTheory.Limits.HasZeroObject C
                           inst✝⁸ : CategoryTheory.Limits.HasZeroObject D
                           inst✝⁷ : CategoryTheory.Preadditive C
                           inst✝⁶ : CategoryTheory.Preadditive D
                           inst✝⁵ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
                           inst✝⁴ : ∀ (n : Int), (CategoryTheory.shiftFunctor D n).Additive
                           inst✝³ : CategoryTheory.Pretriangulated C
                           inst✝² : CategoryTheory.Pretriangulated D
                           X₁ X₂ X₃ Z₁₂ Z₂₃ Z₁₃ : C
                           u₁₂ : Quiver.Hom X₁ X₂
                           u₂₃ : Quiver.Hom X₂ X₃
                           u₁₃ : Quiver.Hom X₁ X₃
                           comm : Eq (CategoryTheory.CategoryStruct.comp u₁₂ u₂₃) u₁₃
                           v₁₂ : Quiver.Hom X₂ Z₁₂
                           w₁₂ : Quiver.Hom Z₁₂ ((CategoryTheory.shiftFunctor C 1).obj X₁)
                           h₁₂ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
                           v₂₃ : Quiver.Hom X₃ Z₂₃
                           w₂₃ : Quiver.Hom Z₂₃ ((CategoryTheory.shiftFunctor C 1).obj X₂)
                           h₂₃ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
                           v₁₃ : Quiver.Hom X₃ Z₁₃
                           w₁₃ : Quiver.Hom Z₁₃ ((CategoryTheory.shiftFunctor C 1).obj X₁)
                           h₁₃ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
                           h : CategoryTheory.Triangulated.Octahedron comm h₁₂ h₂₃ h₁₃
                           F : CategoryTheory.Functor C D
                           inst✝¹ : F.CommShift Int
                           inst✝ : F.IsTriangulated
                           ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map (CategoryTheory.Pretriangulate …
                         -/
def map : Octahedron (by dsimp; rw [← F.map_comp, comm])
                                /-
                                  🎉 no goals
                                -/
    (F.map_distinguished _ h₁₂) (F.map_distinguished _ h₂₃) (F.map_distinguished _ h₁₃) where
  m₁ := F.map h.m₁
  m₃ := F.map h.m₃
              /-
                C : Type u_1
                D : Type u_2
                inst✝¹³ : CategoryTheory.Category.{?u.380803, u_1} C
                inst✝¹² : CategoryTheory.Category.{?u.380807, u_2} D
                inst✝¹¹ : CategoryTheory.HasShift C Int
                inst✝¹⁰ : CategoryTheory.HasShift D Int
                inst✝⁹ : CategoryTheory.Limits.HasZeroObject C
                inst✝⁸ : CategoryTheory.Limits.HasZeroObject D
                inst✝⁷ : CategoryTheory.Preadditive C
                inst✝⁶ : CategoryTheory.Preadditive D
                inst✝⁵ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
                inst✝⁴ : ∀ (n : Int), (CategoryTheory.shiftFunctor D n).Additive
                inst✝³ : CategoryTheory.Pretriangulated C
                inst✝² : CategoryTheory.Pretriangulated D
                X₁ X₂ X₃ Z₁₂ Z₂₃ Z₁₃ : C
                u₁₂ : Quiver.Hom X₁ X₂
                u₂₃ : Quiver.Hom X₂ X₃
                u₁₃ : Quiver.Hom X₁ X₃
                comm : Eq (CategoryTheory.CategoryStruct.comp u₁₂ u₂₃) u₁₃
                v₁₂ : Quiver.Hom X₂ Z₁₂
                w₁₂ : Quiver.Hom Z₁₂ ((CategoryTheory.shiftFunctor C 1).obj X₁)
                h₁₂ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
                v₂₃ : Quiver.Hom X₃ Z₂₃
                w₂₃ : Quiver.Hom Z₂₃ ((CategoryTheory.shiftFunctor C 1).obj X₂)
                h₂₃ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
                v₁₃ : Quiver.Hom X₃ Z₁₃
                w₁₃ : Quiver.Hom Z₁₃ ((CategoryTheory.shiftFunctor C 1).obj X₁)
                h₁₃ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
                h : CategoryTheory.Triangulated.Octahedron comm h₁₂ h₂₃ h₁₃
                F : CategoryTheory.Functor C D
                inst✝¹ : F.CommShift Int
                inst✝ : F.IsTriangulated
                ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map (CategoryTheory.Pretriangulate …
              -/
  comm₁ := by simpa using F.congr_map h.comm₁
              /-
                🎉 no goals
              -/
              /-
                C : Type u_1
                D : Type u_2
                inst✝¹³ : CategoryTheory.Category.{?u.380803, u_1} C
                inst✝¹² : CategoryTheory.Category.{?u.380807, u_2} D
                inst✝¹¹ : CategoryTheory.HasShift C Int
                inst✝¹⁰ : CategoryTheory.HasShift D Int
                inst✝⁹ : CategoryTheory.Limits.HasZeroObject C
                inst✝⁸ : CategoryTheory.Limits.HasZeroObject D
                inst✝⁷ : CategoryTheory.Preadditive C
                inst✝⁶ : CategoryTheory.Preadditive D
                inst✝⁵ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
                inst✝⁴ : ∀ (n : Int), (CategoryTheory.shiftFunctor D n).Additive
                inst✝³ : CategoryTheory.Pretriangulated C
                inst✝² : CategoryTheory.Pretriangulated D
                X₁ X₂ X₃ Z₁₂ Z₂₃ Z₁₃ : C
                u₁₂ : Quiver.Hom X₁ X₂
                u₂₃ : Quiver.Hom X₂ X₃
                u₁₃ : Quiver.Hom X₁ X₃
                comm : Eq (CategoryTheory.CategoryStruct.comp u₁₂ u₂₃) u₁₃
                v₁₂ : Quiver.Hom X₂ Z₁₂
                w₁₂ : Quiver.Hom Z₁₂ ((CategoryTheory.shiftFunctor C 1).obj X₁)
                h₁₂ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
                v₂₃ : Quiver.Hom X₃ Z₂₃
                w₂₃ : Quiver.Hom Z₂₃ ((CategoryTheory.shiftFunctor C 1).obj X₂)
                h₂₃ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
                v₁₃ : Quiver.Hom X₃ Z₁₃
                w₁₃ : Quiver.Hom Z₁₃ ((CategoryTheory.shiftFunctor C 1).obj X₁)
                h₁₃ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
                h : CategoryTheory.Triangulated.Octahedron comm h₁₂ h₂₃ h₁₃
                F : CategoryTheory.Functor C D
                inst✝¹ : F.CommShift Int
                inst✝ : F.IsTriangulated
                ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map h.m₁) (CategoryTheory.Category …
              -/
  comm₂ := by simpa using F.congr_map h.comm₂ =≫ (F.commShiftIso 1).hom.app X₁
              /-
                🎉 no goals
              -/
              /-
                C : Type u_1
                D : Type u_2
                inst✝¹³ : CategoryTheory.Category.{?u.380803, u_1} C
                inst✝¹² : CategoryTheory.Category.{?u.380807, u_2} D
                inst✝¹¹ : CategoryTheory.HasShift C Int
                inst✝¹⁰ : CategoryTheory.HasShift D Int
                inst✝⁹ : CategoryTheory.Limits.HasZeroObject C
                inst✝⁸ : CategoryTheory.Limits.HasZeroObject D
                inst✝⁷ : CategoryTheory.Preadditive C
                inst✝⁶ : CategoryTheory.Preadditive D
                inst✝⁵ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
                inst✝⁴ : ∀ (n : Int), (CategoryTheory.shiftFunctor D n).Additive
                inst✝³ : CategoryTheory.Pretriangulated C
                inst✝² : CategoryTheory.Pretriangulated D
                X₁ X₂ X₃ Z₁₂ Z₂₃ Z₁₃ : C
                u₁₂ : Quiver.Hom X₁ X₂
                u₂₃ : Quiver.Hom X₂ X₃
                u₁₃ : Quiver.Hom X₁ X₃
                comm : Eq (CategoryTheory.CategoryStruct.comp u₁₂ u₂₃) u₁₃
                v₁₂ : Quiver.Hom X₂ Z₁₂
                w₁₂ : Quiver.Hom Z₁₂ ((CategoryTheory.shiftFunctor C 1).obj X₁)
                h₁₂ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
                v₂₃ : Quiver.Hom X₃ Z₂₃
                w₂₃ : Quiver.Hom Z₂₃ ((CategoryTheory.shiftFunctor C 1).obj X₂)
                h₂₃ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
                v₁₃ : Quiver.Hom X₃ Z₁₃
                w₁₃ : Quiver.Hom Z₁₃ ((CategoryTheory.shiftFunctor C 1).obj X₁)
                h₁₃ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
                h : CategoryTheory.Triangulated.Octahedron comm h₁₂ h₂₃ h₁₃
                F : CategoryTheory.Functor C D
                inst✝¹ : F.CommShift Int
                inst✝ : F.IsTriangulated
                ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map (CategoryTheory.Pretriangulate …
              -/
  comm₃ := by simpa using F.congr_map h.comm₃
              /-
                🎉 no goals
              -/
              /-
                C : Type u_1
                D : Type u_2
                inst✝¹³ : CategoryTheory.Category.{?u.380803, u_1} C
                inst✝¹² : CategoryTheory.Category.{?u.380807, u_2} D
                inst✝¹¹ : CategoryTheory.HasShift C Int
                inst✝¹⁰ : CategoryTheory.HasShift D Int
                inst✝⁹ : CategoryTheory.Limits.HasZeroObject C
                inst✝⁸ : CategoryTheory.Limits.HasZeroObject D
                inst✝⁷ : CategoryTheory.Preadditive C
                inst✝⁶ : CategoryTheory.Preadditive D
                inst✝⁵ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
                inst✝⁴ : ∀ (n : Int), (CategoryTheory.shiftFunctor D n).Additive
                inst✝³ : CategoryTheory.Pretriangulated C
                inst✝² : CategoryTheory.Pretriangulated D
                X₁ X₂ X₃ Z₁₂ Z₂₃ Z₁₃ : C
                u₁₂ : Quiver.Hom X₁ X₂
                u₂₃ : Quiver.Hom X₂ X₃
                u₁₃ : Quiver.Hom X₁ X₃
                comm : Eq (CategoryTheory.CategoryStruct.comp u₁₂ u₂₃) u₁₃
                v₁₂ : Quiver.Hom X₂ Z₁₂
                w₁₂ : Quiver.Hom Z₁₂ ((CategoryTheory.shiftFunctor C 1).obj X₁)
                h₁₂ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
                v₂₃ : Quiver.Hom X₃ Z₂₃
                w₂₃ : Quiver.Hom Z₂₃ ((CategoryTheory.shiftFunctor C 1).obj X₂)
                h₂₃ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
                v₁₃ : Quiver.Hom X₃ Z₁₃
                w₁₃ : Quiver.Hom Z₁₃ ((CategoryTheory.shiftFunctor C 1).obj X₁)
                h₁₃ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
                h : CategoryTheory.Triangulated.Octahedron comm h₁₂ h₂₃ h₁₃
                F : CategoryTheory.Functor C D
                inst✝¹ : F.CommShift Int
                inst✝ : F.IsTriangulated
                ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
              -/
  comm₄ := by simpa using F.congr_map h.comm₄ =≫ (F.commShiftIso 1).hom.app X₂
              /-
                🎉 no goals
              -/
  mem := isomorphic_distinguished _ (F.map_distinguished _ h.mem) _
     /-
       C : Type u_1
       D : Type u_2
       inst✝¹³ : CategoryTheory.Category.{?u.380803, u_1} C
       inst✝¹² : CategoryTheory.Category.{?u.380807, u_2} D
       inst✝¹¹ : CategoryTheory.HasShift C Int
       inst✝¹⁰ : CategoryTheory.HasShift D Int
       inst✝⁹ : CategoryTheory.Limits.HasZeroObject C
       inst✝⁸ : CategoryTheory.Limits.HasZeroObject D
       inst✝⁷ : CategoryTheory.Preadditive C
       inst✝⁶ : CategoryTheory.Preadditive D
       inst✝⁵ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
       inst✝⁴ : ∀ (n : Int), (CategoryTheory.shiftFunctor D n).Additive
       inst✝³ : CategoryTheory.Pretriangulated C
       inst✝² : CategoryTheory.Pretriangulated D
       X₁ X₂ X₃ Z₁₂ Z₂₃ Z₁₃ : C
       u₁₂ : Quiver.Hom X₁ X₂
       u₂₃ : Quiver.Hom X₂ X₃
       u₁₃ : Quiver.Hom X₁ X₃
       comm : Eq (CategoryTheory.CategoryStruct.comp u₁₂ u₂₃) u₁₃
       v₁₂ : Quiver.Hom X₂ Z₁₂
       w₁₂ : Quiver.Hom Z₁₂ ((CategoryTheory.shiftFunctor C 1).obj X₁)
       h₁₂ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
       v₂₃ : Quiver.Hom X₃ Z₂₃
       w₂₃ : Quiver.Hom Z₂₃ ((CategoryTheory.shiftFunctor C 1).obj X₂)
       h₂₃ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
       v₁₃ : Quiver.Hom X₃ Z₁₃
       w₁₃ : Quiver.Hom Z₁₃ ((CategoryTheory.shiftFunctor C 1).obj X₁)
       h₁₃ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
       h : CategoryTheory.Triangulated.Octahedron comm h₁₂ h₂₃ h₁₃
       F : CategoryTheory.Functor C D
       inst✝¹ : F.CommShift Int
       inst✝ : F.IsTriangulated
       ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Pretriangulated.Trian …
     -/
     /-
       🎉 no goals
     -/
     /-
       🎉 no goals
     -/
    (Triangle.isoMk _ _ (Iso.refl _) (Iso.refl _) (Iso.refl _))
     /-
       🎉 no goals
     -/


/-- If `F : C ⥤ D` is a triangulated functor from a triangulated category, then `D`
is also triangulated if tuples of composables arrows in `D` can be lifted to `C`. -/
lemma isTriangulated_of_essSurj_mapComposableArrows_two
    (F : C ⥤ D) [F.CommShift ℤ] [F.IsTriangulated]
    [(F.mapComposableArrows 2).EssSurj] [IsTriangulated C] :
    IsTriangulated D := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝¹⁵ : CategoryTheory.Category.{u_3, u_1} C
    inst✝¹⁴ : CategoryTheory.Category.{u_4, u_2} D
    inst✝¹³ : CategoryTheory.HasShift C Int
    inst✝¹² : CategoryTheory.HasShift D Int
    inst✝¹¹ : CategoryTheory.Limits.HasZeroObject C
    inst✝¹⁰ : CategoryTheory.Limits.HasZeroObject D
    inst✝⁹ : CategoryTheory.Preadditive C
    inst✝⁸ : CategoryTheory.Preadditive D
    inst✝⁷ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝⁶ : ∀ (n : Int), (CategoryTheory.shiftFunctor D n).Additive
    inst✝⁵ : CategoryTheory.Pretriangulated C
    inst✝⁴ : CategoryTheory.Pretriangulated D
    F : CategoryTheory.Functor C D
    inst✝³ : F.CommShift Int
    inst✝² : F.IsTriangulated
    inst✝¹ : (F.mapComposableArrows 2).EssSurj
    inst✝ : CategoryTheory.IsTriangulated C
    ⊢ CategoryTheory.IsTriangulated D
  -/
  apply IsTriangulated.mk
  /-
    case octahedron_axiom
    C : Type u_1
    D : Type u_2
    inst✝¹⁵ : CategoryTheory.Category.{u_3, u_1} C
    inst✝¹⁴ : CategoryTheory.Category.{u_4, u_2} D
    inst✝¹³ : CategoryTheory.HasShift C Int
    inst✝¹² : CategoryTheory.HasShift D Int
    inst✝¹¹ : CategoryTheory.Limits.HasZeroObject C
    inst✝¹⁰ : CategoryTheory.Limits.HasZeroObject D
    inst✝⁹ : CategoryTheory.Preadditive C
    inst✝⁸ : CategoryTheory.Preadditive D
    inst✝⁷ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝⁶ : ∀ (n : Int), (CategoryTheory.shiftFunctor D n).Additive
    inst✝⁵ : CategoryTheory.Pretriangulated C
    inst✝⁴ : CategoryTheory.Pretriangulated D
    F : CategoryTheory.Functor C D
    inst✝³ : F.CommShift Int
    inst✝² : F.IsTriangulated
    inst✝¹ : (F.mapComposableArrows 2).EssSurj
    inst✝ : CategoryTheory.IsTriangulated C
    ⊢ ∀ {X₁ X₂ X₃ Z₁₂ Z₂₃ Z₁₃ : D} {u₁₂ : Quiver.Hom X₁ X₂} {u₂₃ : Quiver.Hom X₂ X …
  -/
  intro Y₁ Y₂ Y₃ Z₁₂ Z₂₃ Z₁₃ u₁₂ u₂₃ u₁₃ comm v₁₂ w₁₂ h₁₂ v₂₃ w₂₃ h₂₃ v₁₃ w₁₃ h₁₃
  obtain ⟨α, ⟨e⟩⟩ : ∃ (α : ComposableArrows C 2),
      Nonempty ((F.mapComposableArrows 2).obj α ≅ ComposableArrows.mk₂ u₁₂ u₂₃) :=
    ⟨_, ⟨Functor.objObjPreimageIso _ _⟩⟩
  /-
    case octahedron_axiom.intro.intro
    C : Type u_1
    D : Type u_2
    inst✝¹⁵ : CategoryTheory.Category.{u_3, u_1} C
    inst✝¹⁴ : CategoryTheory.Category.{u_4, u_2} D
    inst✝¹³ : CategoryTheory.HasShift C Int
    inst✝¹² : CategoryTheory.HasShift D Int
    inst✝¹¹ : CategoryTheory.Limits.HasZeroObject C
    inst✝¹⁰ : CategoryTheory.Limits.HasZeroObject D
    inst✝⁹ : CategoryTheory.Preadditive C
    inst✝⁸ : CategoryTheory.Preadditive D
    inst✝⁷ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝⁶ : ∀ (n : Int), (CategoryTheory.shiftFunctor D n).Additive
    inst✝⁵ : CategoryTheory.Pretriangulated C
    inst✝⁴ : CategoryTheory.Pretriangulated D
    F : CategoryTheory.Functor C D
    inst✝³ : F.CommShift Int
    inst✝² : F.IsTriangulated
    inst✝¹ : (F.mapComposableArrows 2).EssSurj
    inst✝ : CategoryTheory.IsTriangulated C
    Y₁ Y₂ Y₃ Z₁₂ Z₂₃ Z₁₃ : D
    u₁₂ : Quiver.Hom Y₁ Y₂
    u₂₃ : Quiver.Hom Y₂ Y₃
    u₁₃ : Quiver.Hom Y₁ Y₃
    comm : Eq (CategoryTheory.CategoryStruct.comp u₁₂ u₂₃) u₁₃
    v₁₂ : Quiver.Hom Y₂ Z₁₂
    w₁₂ : Quiver.Hom Z₁₂ ((CategoryTheory.shiftFunctor D 1).obj Y₁)
    h₁₂ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
    v₂₃ : Quiver.Hom Y₃ Z₂₃
    w₂₃ : Quiver.Hom Z₂₃ ((CategoryTheory.shiftFunctor D 1).obj Y₂)
    h₂₃ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
    v₁₃ : Quiver.Hom Y₃ Z₁₃
    w₁₃ : Quiver.Hom Z₁₃ ((CategoryTheory.shiftFunctor D 1).obj Y₁)
    h₁₃ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
    α : CategoryTheory.ComposableArrows C 2
    e : CategoryTheory.Iso ((F.mapComposableArrows 2).obj α) (CategoryTheory.Compo …
    ⊢ Nonempty (CategoryTheory.Triangulated.Octahedron comm h₁₂ h₂₃ h₁₃)
  -/
  obtain ⟨X₁, X₂, X₃, f, g, rfl⟩ := ComposableArrows.mk₂_surjective α
  /-
    case octahedron_axiom.intro.intro.intro.intro.intro.intro.intro
    C : Type u_1
    D : Type u_2
    inst✝¹⁵ : CategoryTheory.Category.{u_3, u_1} C
    inst✝¹⁴ : CategoryTheory.Category.{u_4, u_2} D
    inst✝¹³ : CategoryTheory.HasShift C Int
    inst✝¹² : CategoryTheory.HasShift D Int
    inst✝¹¹ : CategoryTheory.Limits.HasZeroObject C
    inst✝¹⁰ : CategoryTheory.Limits.HasZeroObject D
    inst✝⁹ : CategoryTheory.Preadditive C
    inst✝⁸ : CategoryTheory.Preadditive D
    inst✝⁷ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝⁶ : ∀ (n : Int), (CategoryTheory.shiftFunctor D n).Additive
    inst✝⁵ : CategoryTheory.Pretriangulated C
    inst✝⁴ : CategoryTheory.Pretriangulated D
    F : CategoryTheory.Functor C D
    inst✝³ : F.CommShift Int
    inst✝² : F.IsTriangulated
    inst✝¹ : (F.mapComposableArrows 2).EssSurj
    inst✝ : CategoryTheory.IsTriangulated C
    Y₁ Y₂ Y₃ Z₁₂ Z₂₃ Z₁₃ : D
    u₁₂ : Quiver.Hom Y₁ Y₂
    u₂₃ : Quiver.Hom Y₂ Y₃
    u₁₃ : Quiver.Hom Y₁ Y₃
    comm : Eq (CategoryTheory.CategoryStruct.comp u₁₂ u₂₃) u₁₃
    v₁₂ : Quiver.Hom Y₂ Z₁₂
    w₁₂ : Quiver.Hom Z₁₂ ((CategoryTheory.shiftFunctor D 1).obj Y₁)
    h₁₂ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
    v₂₃ : Quiver.Hom Y₃ Z₂₃
    w₂₃ : Quiver.Hom Z₂₃ ((CategoryTheory.shiftFunctor D 1).obj Y₂)
    h₂₃ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
    v₁₃ : Quiver.Hom Y₃ Z₁₃
    w₁₃ : Quiver.Hom Z₁₃ ((CategoryTheory.shiftFunctor D 1).obj Y₁)
    h₁₃ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
    X₁ X₂ X₃ : C
    f : Quiver.Hom X₁ X₂
    g : Quiver.Hom X₂ X₃
    e : CategoryTheory.Iso ((F.mapComposableArrows 2).obj (CategoryTheory.Composab …
    ⊢ Nonempty (CategoryTheory.Triangulated.Octahedron comm h₁₂ h₂₃ h₁₃)
  -/
  obtain ⟨_, _, _, h₁₂'⟩ := distinguished_cocone_triangle f
  /-
    case octahedron_axiom.intro.intro.intro.intro.intro.intro.intro.intro.intro.in …
    C : Type u_1
    D : Type u_2
    inst✝¹⁵ : CategoryTheory.Category.{u_3, u_1} C
    inst✝¹⁴ : CategoryTheory.Category.{u_4, u_2} D
    inst✝¹³ : CategoryTheory.HasShift C Int
    inst✝¹² : CategoryTheory.HasShift D Int
    inst✝¹¹ : CategoryTheory.Limits.HasZeroObject C
    inst✝¹⁰ : CategoryTheory.Limits.HasZeroObject D
    inst✝⁹ : CategoryTheory.Preadditive C
    inst✝⁸ : CategoryTheory.Preadditive D
    inst✝⁷ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝⁶ : ∀ (n : Int), (CategoryTheory.shiftFunctor D n).Additive
    inst✝⁵ : CategoryTheory.Pretriangulated C
    inst✝⁴ : CategoryTheory.Pretriangulated D
    F : CategoryTheory.Functor C D
    inst✝³ : F.CommShift Int
    inst✝² : F.IsTriangulated
    inst✝¹ : (F.mapComposableArrows 2).EssSurj
    inst✝ : CategoryTheory.IsTriangulated C
    Y₁ Y₂ Y₃ Z₁₂ Z₂₃ Z₁₃ : D
    u₁₂ : Quiver.Hom Y₁ Y₂
    u₂₃ : Quiver.Hom Y₂ Y₃
    u₁₃ : Quiver.Hom Y₁ Y₃
    comm : Eq (CategoryTheory.CategoryStruct.comp u₁₂ u₂₃) u₁₃
    v₁₂ : Quiver.Hom Y₂ Z₁₂
    w₁₂ : Quiver.Hom Z₁₂ ((CategoryTheory.shiftFunctor D 1).obj Y₁)
    h₁₂ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
    v₂₃ : Quiver.Hom Y₃ Z₂₃
    w₂₃ : Quiver.Hom Z₂₃ ((CategoryTheory.shiftFunctor D 1).obj Y₂)
    h₂₃ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
    v₁₃ : Quiver.Hom Y₃ Z₁₃
    w₁₃ : Quiver.Hom Z₁₃ ((CategoryTheory.shiftFunctor D 1).obj Y₁)
    h₁₃ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
    X₁ X₂ X₃ : C
    f : Quiver.Hom X₁ X₂
    g : Quiver.Hom X₂ X₃
    e : CategoryTheory.Iso ((F.mapComposableArrows 2).obj (CategoryTheory.Composab …
    w✝² : C
    w✝¹ : Quiver.Hom X₂ w✝²
    w✝ : Quiver.Hom w✝² ((CategoryTheory.shiftFunctor C 1).obj X₁)
    h₁₂' : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (C …
    ⊢ Nonempty (CategoryTheory.Triangulated.Octahedron comm h₁₂ h₂₃ h₁₃)
  -/
  obtain ⟨_, _, _, h₂₃'⟩ := distinguished_cocone_triangle g
  /-
    case octahedron_axiom.intro.intro.intro.intro.intro.intro.intro.intro.intro.in …
    C : Type u_1
    D : Type u_2
    inst✝¹⁵ : CategoryTheory.Category.{u_3, u_1} C
    inst✝¹⁴ : CategoryTheory.Category.{u_4, u_2} D
    inst✝¹³ : CategoryTheory.HasShift C Int
    inst✝¹² : CategoryTheory.HasShift D Int
    inst✝¹¹ : CategoryTheory.Limits.HasZeroObject C
    inst✝¹⁰ : CategoryTheory.Limits.HasZeroObject D
    inst✝⁹ : CategoryTheory.Preadditive C
    inst✝⁸ : CategoryTheory.Preadditive D
    inst✝⁷ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝⁶ : ∀ (n : Int), (CategoryTheory.shiftFunctor D n).Additive
    inst✝⁵ : CategoryTheory.Pretriangulated C
    inst✝⁴ : CategoryTheory.Pretriangulated D
    F : CategoryTheory.Functor C D
    inst✝³ : F.CommShift Int
    inst✝² : F.IsTriangulated
    inst✝¹ : (F.mapComposableArrows 2).EssSurj
    inst✝ : CategoryTheory.IsTriangulated C
    Y₁ Y₂ Y₃ Z₁₂ Z₂₃ Z₁₃ : D
    u₁₂ : Quiver.Hom Y₁ Y₂
    u₂₃ : Quiver.Hom Y₂ Y₃
    u₁₃ : Quiver.Hom Y₁ Y₃
    comm : Eq (CategoryTheory.CategoryStruct.comp u₁₂ u₂₃) u₁₃
    v₁₂ : Quiver.Hom Y₂ Z₁₂
    w₁₂ : Quiver.Hom Z₁₂ ((CategoryTheory.shiftFunctor D 1).obj Y₁)
    h₁₂ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
    v₂₃ : Quiver.Hom Y₃ Z₂₃
    w₂₃ : Quiver.Hom Z₂₃ ((CategoryTheory.shiftFunctor D 1).obj Y₂)
    h₂₃ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
    v₁₃ : Quiver.Hom Y₃ Z₁₃
    w₁₃ : Quiver.Hom Z₁₃ ((CategoryTheory.shiftFunctor D 1).obj Y₁)
    h₁₃ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
    X₁ X₂ X₃ : C
    f : Quiver.Hom X₁ X₂
    g : Quiver.Hom X₂ X₃
    e : CategoryTheory.Iso ((F.mapComposableArrows 2).obj (CategoryTheory.Composab …
    w✝⁵ : C
    w✝⁴ : Quiver.Hom X₂ w✝⁵
    w✝³ : Quiver.Hom w✝⁵ ((CategoryTheory.shiftFunctor C 1).obj X₁)
    h₁₂' : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (C …
    w✝² : C
    w✝¹ : Quiver.Hom X₃ w✝²
    w✝ : Quiver.Hom w✝² ((CategoryTheory.shiftFunctor C 1).obj X₂)
    h₂₃' : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (C …
    ⊢ Nonempty (CategoryTheory.Triangulated.Octahedron comm h₁₂ h₂₃ h₁₃)
  -/
  obtain ⟨_, _, _, h₁₃'⟩ := distinguished_cocone_triangle (f ≫ g)
  exact ⟨Octahedron.ofIso (e₁ := (e.app 0).symm) (e₂ := (e.app 1).symm) (e₃ := (e.app 2).symm)
    (comm₁₂ := ComposableArrows.naturality' e.inv 0 1)
    (comm₂₃ := ComposableArrows.naturality' e.inv 1 2)
    (H := (someOctahedron rfl h₁₂' h₂₃' h₁₃').map F) ..⟩


