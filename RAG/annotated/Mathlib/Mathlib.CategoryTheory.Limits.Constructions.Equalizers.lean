/-- Define the equalizing object -/
abbrev constructEqualizer (F : WalkingParallelPair ⥤ C) : C :=
  pullback (prod.lift (𝟙 _) (F.map WalkingParallelPairHom.left))
    (prod.lift (𝟙 _) (F.map WalkingParallelPairHom.right))


/-- Define the equalizing morphism -/
abbrev pullbackFst (F : WalkingParallelPair ⥤ C) :
    constructEqualizer F ⟶ F.obj WalkingParallelPair.zero :=
  pullback.fst _ _


theorem pullbackFst_eq_pullback_snd (F : WalkingParallelPair ⥤ C) :
    pullbackFst F = pullback.snd _ _ := by
  convert (eq_whisker pullback.condition Limits.prod.fst :
                                                                           /-
                                                                             case h.e'_2
                                                                             C : Type u
                                                                             inst✝² : CategoryTheory.Category.{v, u} C
                                                                             inst✝¹ : CategoryTheory.Limits.HasBinaryProducts C
                                                                             inst✝ : CategoryTheory.Limits.HasPullbacks C
                                                                             F : CategoryTheory.Functor CategoryTheory.Limits.WalkingParallelPair C
                                                                             ⊢ Eq (CategoryTheory.Limits.HasEqualizersOfHasPullbacksAndBinaryProducts.pullb …
                                                                           -/
                                                                           /-
                                                                             🎉 no goals
                                                                           -/
      (_ : constructEqualizer F ⟶ F.obj WalkingParallelPair.zero) = _) <;> simp
                                                                           /-
                                                                             🎉 no goals
                                                                           -/


/-- Define the equalizing cone -/
abbrev equalizerCone (F : WalkingParallelPair ⥤ C) : Cone F :=
  Cone.ofFork
    (Fork.ofι (pullbackFst F)
      (by
        /-
          C : Type u
          inst✝³ : CategoryTheory.Category.{v, u} C
          D : Type u'
          inst✝² : CategoryTheory.Category.{v', u'} D
          G : CategoryTheory.Functor C D
          inst✝¹ : CategoryTheory.Limits.HasBinaryProducts C
          inst✝ : CategoryTheory.Limits.HasPullbacks C
          F : CategoryTheory.Functor CategoryTheory.Limits.WalkingParallelPair C
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.HasEqualizersO …
        -/
        conv_rhs => rw [pullbackFst_eq_pullback_snd]
        convert (eq_whisker pullback.condition Limits.prod.snd :
                                                                                      /-
                                                                                        case h.e'_2
                                                                                        C : Type u
                                                                                        inst✝³ : CategoryTheory.Category.{v, u} C
                                                                                        D : Type u'
                                                                                        inst✝² : CategoryTheory.Category.{v', u'} D
                                                                                        G : CategoryTheory.Functor C D
                                                                                        inst✝¹ : CategoryTheory.Limits.HasBinaryProducts C
                                                                                        inst✝ : CategoryTheory.Limits.HasPullbacks C
                                                                                        F : CategoryTheory.Functor CategoryTheory.Limits.WalkingParallelPair C
                                                                                        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.HasEqualizersO …
                                                                                      -/
                                                                                      /-
                                                                                        🎉 no goals
                                                                                      -/
          (_ : constructEqualizer F ⟶ F.obj WalkingParallelPair.one) = _) using 1 <;> simp))
                                                                                      /-
                                                                                        🎉 no goals
                                                                                      -/


/-- Show the equalizing cone is a limit -/
def equalizerConeIsLimit (F : WalkingParallelPair ⥤ C) : IsLimit (equalizerCone F) where
  lift := by
    /-
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝² : CategoryTheory.Category.{v', u'} D
      G : CategoryTheory.Functor C D
      inst✝¹ : CategoryTheory.Limits.HasBinaryProducts C
      inst✝ : CategoryTheory.Limits.HasPullbacks C
      F : CategoryTheory.Functor CategoryTheory.Limits.WalkingParallelPair C
      ⊢ (s : CategoryTheory.Limits.Cone F) → Quiver.Hom s.pt (CategoryTheory.Limits. …
    -/
    intro c; apply pullback.lift (c.π.app _) (c.π.app _)
    /-
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝² : CategoryTheory.Category.{v', u'} D
      G : CategoryTheory.Functor C D
      inst✝¹ : CategoryTheory.Limits.HasBinaryProducts C
      inst✝ : CategoryTheory.Limits.HasPullbacks C
      F : CategoryTheory.Functor CategoryTheory.Limits.WalkingParallelPair C
      c : CategoryTheory.Limits.Cone F
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (c.π.app CategoryTheory.Limits.Walkin …
    -/
            /-
              🎉 no goals
            -/
    ext <;> simp
            /-
              🎉 no goals
            -/
            /-
              C : Type u
              inst✝³ : CategoryTheory.Category.{v, u} C
              D : Type u'
              inst✝² : CategoryTheory.Category.{v', u'} D
              G : CategoryTheory.Functor C D
              inst✝¹ : CategoryTheory.Limits.HasBinaryProducts C
              inst✝ : CategoryTheory.Limits.HasPullbacks C
              F : CategoryTheory.Functor CategoryTheory.Limits.WalkingParallelPair C
              ⊢ ∀ (s : CategoryTheory.Limits.Cone F) (j : CategoryTheory.Limits.WalkingParal …
            -/
                                 /-
                                   🎉 no goals
                                 -/
  fac := by rintro c (_ | _) <;> simp
                                 /-
                                   🎉 no goals
                                 -/
  uniq := by
    /-
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝² : CategoryTheory.Category.{v', u'} D
      G : CategoryTheory.Functor C D
      inst✝¹ : CategoryTheory.Limits.HasBinaryProducts C
      inst✝ : CategoryTheory.Limits.HasPullbacks C
      F : CategoryTheory.Functor CategoryTheory.Limits.WalkingParallelPair C
      ⊢ ∀ (s : CategoryTheory.Limits.Cone F) (m : Quiver.Hom s.pt (CategoryTheory.Li …
    -/
    intro c _ J
    /-
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝² : CategoryTheory.Category.{v', u'} D
      G : CategoryTheory.Functor C D
      inst✝¹ : CategoryTheory.Limits.HasBinaryProducts C
      inst✝ : CategoryTheory.Limits.HasPullbacks C
      F : CategoryTheory.Functor CategoryTheory.Limits.WalkingParallelPair C
      c : CategoryTheory.Limits.Cone F
      m✝ : Quiver.Hom c.pt (CategoryTheory.Limits.HasEqualizersOfHasPullbacksAndBina …
      J : ∀ (j : CategoryTheory.Limits.WalkingParallelPair), Eq (CategoryTheory.Cate …
      ⊢ Eq m✝ (CategoryTheory.Limits.pullback.lift (c.π.app CategoryTheory.Limits.Wa …
    -/
    have J0 := J WalkingParallelPair.zero; simp at J0
    /-
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝² : CategoryTheory.Category.{v', u'} D
      G : CategoryTheory.Functor C D
      inst✝¹ : CategoryTheory.Limits.HasBinaryProducts C
      inst✝ : CategoryTheory.Limits.HasPullbacks C
      F : CategoryTheory.Functor CategoryTheory.Limits.WalkingParallelPair C
      c : CategoryTheory.Limits.Cone F
      m✝ : Quiver.Hom c.pt (CategoryTheory.Limits.HasEqualizersOfHasPullbacksAndBina …
      J : ∀ (j : CategoryTheory.Limits.WalkingParallelPair), Eq (CategoryTheory.Cate …
      J0 : Eq (CategoryTheory.CategoryStruct.comp m✝ (CategoryTheory.Limits.HasEqual …
      ⊢ Eq m✝ (CategoryTheory.Limits.pullback.lift (c.π.app CategoryTheory.Limits.Wa …
    -/
    apply pullback.hom_ext
      /-
        case h₀
        C : Type u
        inst✝³ : CategoryTheory.Category.{v, u} C
        D : Type u'
        inst✝² : CategoryTheory.Category.{v', u'} D
        G : CategoryTheory.Functor C D
        inst✝¹ : CategoryTheory.Limits.HasBinaryProducts C
        inst✝ : CategoryTheory.Limits.HasPullbacks C
        F : CategoryTheory.Functor CategoryTheory.Limits.WalkingParallelPair C
        c : CategoryTheory.Limits.Cone F
        m✝ : Quiver.Hom c.pt (CategoryTheory.Limits.HasEqualizersOfHasPullbacksAndBina …
        J : ∀ (j : CategoryTheory.Limits.WalkingParallelPair), Eq (CategoryTheory.Cate …
        J0 : Eq (CategoryTheory.CategoryStruct.comp m✝ (CategoryTheory.Limits.HasEqual …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp m✝ (CategoryTheory.Limits.pullback.fs …
      -/
    · rwa [limit.lift_π]
      /-
        🎉 no goals
      -/
      /-
        case h₁
        C : Type u
        inst✝³ : CategoryTheory.Category.{v, u} C
        D : Type u'
        inst✝² : CategoryTheory.Category.{v', u'} D
        G : CategoryTheory.Functor C D
        inst✝¹ : CategoryTheory.Limits.HasBinaryProducts C
        inst✝ : CategoryTheory.Limits.HasPullbacks C
        F : CategoryTheory.Functor CategoryTheory.Limits.WalkingParallelPair C
        c : CategoryTheory.Limits.Cone F
        m✝ : Quiver.Hom c.pt (CategoryTheory.Limits.HasEqualizersOfHasPullbacksAndBina …
        J : ∀ (j : CategoryTheory.Limits.WalkingParallelPair), Eq (CategoryTheory.Cate …
        J0 : Eq (CategoryTheory.CategoryStruct.comp m✝ (CategoryTheory.Limits.HasEqual …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp m✝ (CategoryTheory.Limits.pullback.sn …
      -/
    · erw [limit.lift_π, ← J0, pullbackFst_eq_pullback_snd]
      /-
        🎉 no goals
      -/


/-- Any category with pullbacks and binary products, has equalizers. -/
theorem hasEqualizers_of_hasPullbacks_and_binary_products [HasBinaryProducts C] [HasPullbacks C] :
    HasEqualizers C :=
  { has_limit := fun F =>
      HasLimit.mk
        { cone := equalizerCone F
          isLimit := equalizerConeIsLimit F } }


/-- A functor that preserves pullbacks and binary products also presrves equalizers. -/
lemma preservesEqualizers_of_preservesPullbacks_and_binaryProducts
    [HasBinaryProducts C] [HasPullbacks C]
    [PreservesLimitsOfShape (Discrete WalkingPair) G] [PreservesLimitsOfShape WalkingCospan G] :
    PreservesLimitsOfShape WalkingParallelPair G :=
  ⟨fun {K} =>
    preservesLimit_of_preserves_limit_cone (equalizerConeIsLimit K) <|
      { lift := fun c => by
          /-
            C : Type u
            inst✝⁵ : CategoryTheory.Category.{v, u} C
            D : Type u'
            inst✝⁴ : CategoryTheory.Category.{v', u'} D
            G : CategoryTheory.Functor C D
            inst✝³ : CategoryTheory.Limits.HasBinaryProducts C
            inst✝² : CategoryTheory.Limits.HasPullbacks C
            inst✝¹ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete …
            inst✝ : CategoryTheory.Limits.PreservesLimitsOfShape CategoryTheory.Limits.Wal …
            K : CategoryTheory.Functor CategoryTheory.Limits.WalkingParallelPair C
            c : CategoryTheory.Limits.Cone (K.comp G)
            ⊢ Quiver.Hom c.pt (G.mapCone (CategoryTheory.Limits.HasEqualizersOfHasPullback …
          -/
          refine pullback.lift ?_ ?_ ?_ ≫ (PreservesPullback.iso _ _ _ ).inv
            /-
              case refine_1
              C : Type u
              inst✝⁵ : CategoryTheory.Category.{v, u} C
              D : Type u'
              inst✝⁴ : CategoryTheory.Category.{v', u'} D
              G : CategoryTheory.Functor C D
              inst✝³ : CategoryTheory.Limits.HasBinaryProducts C
              inst✝² : CategoryTheory.Limits.HasPullbacks C
              inst✝¹ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete …
              inst✝ : CategoryTheory.Limits.PreservesLimitsOfShape CategoryTheory.Limits.Wal …
              K : CategoryTheory.Functor CategoryTheory.Limits.WalkingParallelPair C
              c : CategoryTheory.Limits.Cone (K.comp G)
              ⊢ Quiver.Hom c.pt (G.obj (K.obj CategoryTheory.Limits.WalkingParallelPair.zero))
            -/
          · exact c.π.app WalkingParallelPair.zero
            /-
              🎉 no goals
            -/
            /-
              case refine_2
              C : Type u
              inst✝⁵ : CategoryTheory.Category.{v, u} C
              D : Type u'
              inst✝⁴ : CategoryTheory.Category.{v', u'} D
              G : CategoryTheory.Functor C D
              inst✝³ : CategoryTheory.Limits.HasBinaryProducts C
              inst✝² : CategoryTheory.Limits.HasPullbacks C
              inst✝¹ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete …
              inst✝ : CategoryTheory.Limits.PreservesLimitsOfShape CategoryTheory.Limits.Wal …
              K : CategoryTheory.Functor CategoryTheory.Limits.WalkingParallelPair C
              c : CategoryTheory.Limits.Cone (K.comp G)
              ⊢ Quiver.Hom c.pt (G.obj (K.obj CategoryTheory.Limits.WalkingParallelPair.zero))
            -/
          · exact c.π.app WalkingParallelPair.zero
            /-
              🎉 no goals
            -/
          /-
            case refine_3
            C : Type u
            inst✝⁵ : CategoryTheory.Category.{v, u} C
            D : Type u'
            inst✝⁴ : CategoryTheory.Category.{v', u'} D
            G : CategoryTheory.Functor C D
            inst✝³ : CategoryTheory.Limits.HasBinaryProducts C
            inst✝² : CategoryTheory.Limits.HasPullbacks C
            inst✝¹ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete …
            inst✝ : CategoryTheory.Limits.PreservesLimitsOfShape CategoryTheory.Limits.Wal …
            K : CategoryTheory.Functor CategoryTheory.Limits.WalkingParallelPair C
            c : CategoryTheory.Limits.Cone (K.comp G)
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (c.π.app CategoryTheory.Limits.Walkin …
          -/
          apply (mapIsLimitOfPreservesOfIsLimit G _ _ (prodIsProd _ _)).hom_ext
          /-
            case refine_3
            C : Type u
            inst✝⁵ : CategoryTheory.Category.{v, u} C
            D : Type u'
            inst✝⁴ : CategoryTheory.Category.{v', u'} D
            G : CategoryTheory.Functor C D
            inst✝³ : CategoryTheory.Limits.HasBinaryProducts C
            inst✝² : CategoryTheory.Limits.HasPullbacks C
            inst✝¹ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete …
            inst✝ : CategoryTheory.Limits.PreservesLimitsOfShape CategoryTheory.Limits.Wal …
            K : CategoryTheory.Functor CategoryTheory.Limits.WalkingParallelPair C
            c : CategoryTheory.Limits.Cone (K.comp G)
            ⊢ ∀ (j : CategoryTheory.Discrete CategoryTheory.Limits.WalkingPair), Eq (Categ …
          -/
          rintro (_ | _)
          · simp only [Category.assoc, ← G.map_comp, prod.lift_fst, BinaryFan.π_app_left,
              BinaryFan.mk_fst]
          · simp only [BinaryFan.π_app_right, BinaryFan.mk_snd, Category.assoc, ← G.map_comp,
              prod.lift_snd]
            exact
              (c.π.naturality WalkingParallelPairHom.left).symm.trans
                (c.π.naturality WalkingParallelPairHom.right)
        fac := fun c j => by
          /-
            C : Type u
            inst✝⁵ : CategoryTheory.Category.{v, u} C
            D : Type u'
            inst✝⁴ : CategoryTheory.Category.{v', u'} D
            G : CategoryTheory.Functor C D
            inst✝³ : CategoryTheory.Limits.HasBinaryProducts C
            inst✝² : CategoryTheory.Limits.HasPullbacks C
            inst✝¹ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete …
            inst✝ : CategoryTheory.Limits.PreservesLimitsOfShape CategoryTheory.Limits.Wal …
            K : CategoryTheory.Functor CategoryTheory.Limits.WalkingParallelPair C
            c : CategoryTheory.Limits.Cone (K.comp G)
            j : CategoryTheory.Limits.WalkingParallelPair
            ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun c => CategoryTheory.CategoryStr …
          -/
          rcases j with (_ | _) <;>
            simp only [Category.comp_id, PreservesPullback.iso_inv_fst, Cone.ofFork_π, G.map_comp,
              PreservesPullback.iso_inv_fst_assoc, Functor.mapCone_π_app, eqToHom_refl,
              Category.assoc, Fork.ofι_π_app, pullback.lift_fst, pullback.lift_fst_assoc]
          /-
            case one
            C : Type u
            inst✝⁵ : CategoryTheory.Category.{v, u} C
            D : Type u'
            inst✝⁴ : CategoryTheory.Category.{v', u'} D
            G : CategoryTheory.Functor C D
            inst✝³ : CategoryTheory.Limits.HasBinaryProducts C
            inst✝² : CategoryTheory.Limits.HasPullbacks C
            inst✝¹ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete …
            inst✝ : CategoryTheory.Limits.PreservesLimitsOfShape CategoryTheory.Limits.Wal …
            K : CategoryTheory.Functor CategoryTheory.Limits.WalkingParallelPair C
            c : CategoryTheory.Limits.Cone (K.comp G)
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (c.π.app CategoryTheory.Limits.Walkin …
          -/
          exact (c.π.naturality WalkingParallelPairHom.left).symm.trans (Category.id_comp _)
          /-
            🎉 no goals
          -/
        uniq := fun s m h => by
          /-
            C : Type u
            inst✝⁵ : CategoryTheory.Category.{v, u} C
            D : Type u'
            inst✝⁴ : CategoryTheory.Category.{v', u'} D
            G : CategoryTheory.Functor C D
            inst✝³ : CategoryTheory.Limits.HasBinaryProducts C
            inst✝² : CategoryTheory.Limits.HasPullbacks C
            inst✝¹ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete …
            inst✝ : CategoryTheory.Limits.PreservesLimitsOfShape CategoryTheory.Limits.Wal …
            K : CategoryTheory.Functor CategoryTheory.Limits.WalkingParallelPair C
            s : CategoryTheory.Limits.Cone (K.comp G)
            m : Quiver.Hom s.pt (G.mapCone (CategoryTheory.Limits.HasEqualizersOfHasPullba …
            h : ∀ (j : CategoryTheory.Limits.WalkingParallelPair), Eq (CategoryTheory.Cate …
            ⊢ Eq m ((fun c => CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pu …
          -/
          rw [Iso.eq_comp_inv]
          /-
            C : Type u
            inst✝⁵ : CategoryTheory.Category.{v, u} C
            D : Type u'
            inst✝⁴ : CategoryTheory.Category.{v', u'} D
            G : CategoryTheory.Functor C D
            inst✝³ : CategoryTheory.Limits.HasBinaryProducts C
            inst✝² : CategoryTheory.Limits.HasPullbacks C
            inst✝¹ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete …
            inst✝ : CategoryTheory.Limits.PreservesLimitsOfShape CategoryTheory.Limits.Wal …
            K : CategoryTheory.Functor CategoryTheory.Limits.WalkingParallelPair C
            s : CategoryTheory.Limits.Cone (K.comp G)
            m : Quiver.Hom s.pt (G.mapCone (CategoryTheory.Limits.HasEqualizersOfHasPullba …
            h : ∀ (j : CategoryTheory.Limits.WalkingParallelPair), Eq (CategoryTheory.Cate …
            ⊢ Eq (CategoryTheory.CategoryStruct.comp m (CategoryTheory.Limits.PreservesPul …
          -/
          have := h WalkingParallelPair.zero
          /-
            C : Type u
            inst✝⁵ : CategoryTheory.Category.{v, u} C
            D : Type u'
            inst✝⁴ : CategoryTheory.Category.{v', u'} D
            G : CategoryTheory.Functor C D
            inst✝³ : CategoryTheory.Limits.HasBinaryProducts C
            inst✝² : CategoryTheory.Limits.HasPullbacks C
            inst✝¹ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete …
            inst✝ : CategoryTheory.Limits.PreservesLimitsOfShape CategoryTheory.Limits.Wal …
            K : CategoryTheory.Functor CategoryTheory.Limits.WalkingParallelPair C
            s : CategoryTheory.Limits.Cone (K.comp G)
            m : Quiver.Hom s.pt (G.mapCone (CategoryTheory.Limits.HasEqualizersOfHasPullba …
            h : ∀ (j : CategoryTheory.Limits.WalkingParallelPair), Eq (CategoryTheory.Cate …
            this : Eq (CategoryTheory.CategoryStruct.comp m ((G.mapCone (CategoryTheory.Li …
            ⊢ Eq (CategoryTheory.CategoryStruct.comp m (CategoryTheory.Limits.PreservesPul …
          -/
          dsimp [equalizerCone] at this
          /-
            C : Type u
            inst✝⁵ : CategoryTheory.Category.{v, u} C
            D : Type u'
            inst✝⁴ : CategoryTheory.Category.{v', u'} D
            G : CategoryTheory.Functor C D
            inst✝³ : CategoryTheory.Limits.HasBinaryProducts C
            inst✝² : CategoryTheory.Limits.HasPullbacks C
            inst✝¹ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete …
            inst✝ : CategoryTheory.Limits.PreservesLimitsOfShape CategoryTheory.Limits.Wal …
            K : CategoryTheory.Functor CategoryTheory.Limits.WalkingParallelPair C
            s : CategoryTheory.Limits.Cone (K.comp G)
            m : Quiver.Hom s.pt (G.mapCone (CategoryTheory.Limits.HasEqualizersOfHasPullba …
            h : ∀ (j : CategoryTheory.Limits.WalkingParallelPair), Eq (CategoryTheory.Cate …
            this : Eq (CategoryTheory.CategoryStruct.comp m (G.map (CategoryTheory.Categor …
            ⊢ Eq (CategoryTheory.CategoryStruct.comp m (CategoryTheory.Limits.PreservesPul …
          -/
          ext <;>
            simp only [PreservesPullback.iso_hom_snd, Category.assoc,
              PreservesPullback.iso_hom_fst, pullback.lift_fst, pullback.lift_snd,
              Category.comp_id, ← pullbackFst_eq_pullback_snd, ← this] }⟩

-- We hide the "implementation details" inside a namespace

/-- Define the equalizing object -/
abbrev constructCoequalizer (F : WalkingParallelPair ⥤ C) : C :=
  pushout (coprod.desc (𝟙 _) (F.map WalkingParallelPairHom.left))
    (coprod.desc (𝟙 _) (F.map WalkingParallelPairHom.right))


/-- Define the equalizing morphism -/
abbrev pushoutInl (F : WalkingParallelPair ⥤ C) :
    F.obj WalkingParallelPair.one ⟶ constructCoequalizer F :=
  pushout.inl _ _


theorem pushoutInl_eq_pushout_inr (F : WalkingParallelPair ⥤ C) :
    pushoutInl F = pushout.inr _ _ := by
  convert (whisker_eq Limits.coprod.inl pushout.condition :
                                                    /-
                                                      case h.e'_2
                                                      C : Type u
                                                      inst✝² : CategoryTheory.Category.{v, u} C
                                                      inst✝¹ : CategoryTheory.Limits.HasBinaryCoproducts C
                                                      inst✝ : CategoryTheory.Limits.HasPushouts C
                                                      F : CategoryTheory.Functor CategoryTheory.Limits.WalkingParallelPair C
                                                      ⊢ Eq (CategoryTheory.Limits.HasCoequalizersOfHasPushoutsAndBinaryCoproducts.pu …
                                                    -/
                                                    /-
                                                      🎉 no goals
                                                    -/
    (_ : F.obj _ ⟶ constructCoequalizer _) = _) <;> simp
                                                    /-
                                                      🎉 no goals
                                                    -/


/-- Define the equalizing cocone -/
abbrev coequalizerCocone (F : WalkingParallelPair ⥤ C) : Cocone F :=
  Cocone.ofCofork
    (Cofork.ofπ (pushoutInl F) (by
        /-
          C : Type u
          inst✝³ : CategoryTheory.Category.{v, u} C
          D : Type u'
          inst✝² : CategoryTheory.Category.{v', u'} D
          G : CategoryTheory.Functor C D
          inst✝¹ : CategoryTheory.Limits.HasBinaryCoproducts C
          inst✝ : CategoryTheory.Limits.HasPushouts C
          F : CategoryTheory.Functor CategoryTheory.Limits.WalkingParallelPair C
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map CategoryTheory.Limits.WalkingP …
        -/
        conv_rhs => rw [pushoutInl_eq_pushout_inr]
        convert (whisker_eq Limits.coprod.inr pushout.condition :
                                                                  /-
                                                                    case h.e'_2
                                                                    C : Type u
                                                                    inst✝³ : CategoryTheory.Category.{v, u} C
                                                                    D : Type u'
                                                                    inst✝² : CategoryTheory.Category.{v', u'} D
                                                                    G : CategoryTheory.Functor C D
                                                                    inst✝¹ : CategoryTheory.Limits.HasBinaryCoproducts C
                                                                    inst✝ : CategoryTheory.Limits.HasPushouts C
                                                                    F : CategoryTheory.Functor CategoryTheory.Limits.WalkingParallelPair C
                                                                    ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map CategoryTheory.Limits.WalkingP …
                                                                  -/
                                                                  /-
                                                                    🎉 no goals
                                                                  -/
          (_ : F.obj _ ⟶ constructCoequalizer _) = _) using 1 <;> simp))
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


/-- Show the equalizing cocone is a colimit -/
def coequalizerCoconeIsColimit (F : WalkingParallelPair ⥤ C) : IsColimit (coequalizerCocone F) where
  desc := by
    /-
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝² : CategoryTheory.Category.{v', u'} D
      G : CategoryTheory.Functor C D
      inst✝¹ : CategoryTheory.Limits.HasBinaryCoproducts C
      inst✝ : CategoryTheory.Limits.HasPushouts C
      F : CategoryTheory.Functor CategoryTheory.Limits.WalkingParallelPair C
      ⊢ (s : CategoryTheory.Limits.Cocone F) → Quiver.Hom (CategoryTheory.Limits.Has …
    -/
    intro c; apply pushout.desc (c.ι.app _) (c.ι.app _)
    /-
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝² : CategoryTheory.Category.{v', u'} D
      G : CategoryTheory.Functor C D
      inst✝¹ : CategoryTheory.Limits.HasBinaryCoproducts C
      inst✝ : CategoryTheory.Limits.HasPushouts C
      F : CategoryTheory.Functor CategoryTheory.Limits.WalkingParallelPair C
      c : CategoryTheory.Limits.Cocone F
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.coprod.desc (C …
    -/
            /-
              🎉 no goals
            -/
    ext <;> simp
            /-
              🎉 no goals
            -/
            /-
              C : Type u
              inst✝³ : CategoryTheory.Category.{v, u} C
              D : Type u'
              inst✝² : CategoryTheory.Category.{v', u'} D
              G : CategoryTheory.Functor C D
              inst✝¹ : CategoryTheory.Limits.HasBinaryCoproducts C
              inst✝ : CategoryTheory.Limits.HasPushouts C
              F : CategoryTheory.Functor CategoryTheory.Limits.WalkingParallelPair C
              ⊢ ∀ (s : CategoryTheory.Limits.Cocone F) (j : CategoryTheory.Limits.WalkingPar …
            -/
                                 /-
                                   🎉 no goals
                                 -/
  fac := by rintro c (_ | _) <;> simp
                                 /-
                                   🎉 no goals
                                 -/
  uniq := by
    /-
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝² : CategoryTheory.Category.{v', u'} D
      G : CategoryTheory.Functor C D
      inst✝¹ : CategoryTheory.Limits.HasBinaryCoproducts C
      inst✝ : CategoryTheory.Limits.HasPushouts C
      F : CategoryTheory.Functor CategoryTheory.Limits.WalkingParallelPair C
      ⊢ ∀ (s : CategoryTheory.Limits.Cocone F) (m : Quiver.Hom (CategoryTheory.Limit …
    -/
    intro c m J
    have J1 : pushoutInl F ≫ m = c.ι.app WalkingParallelPair.one := by
      simpa using J WalkingParallelPair.one
    /-
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝² : CategoryTheory.Category.{v', u'} D
      G : CategoryTheory.Functor C D
      inst✝¹ : CategoryTheory.Limits.HasBinaryCoproducts C
      inst✝ : CategoryTheory.Limits.HasPushouts C
      F : CategoryTheory.Functor CategoryTheory.Limits.WalkingParallelPair C
      c : CategoryTheory.Limits.Cocone F
      m : Quiver.Hom (CategoryTheory.Limits.HasCoequalizersOfHasPushoutsAndBinaryCop …
      J : ∀ (j : CategoryTheory.Limits.WalkingParallelPair), Eq (CategoryTheory.Cate …
      J1 : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.HasCoequali …
      ⊢ Eq m (CategoryTheory.Limits.pushout.desc (c.ι.app CategoryTheory.Limits.Walk …
    -/
    apply pushout.hom_ext
      /-
        case h₀
        C : Type u
        inst✝³ : CategoryTheory.Category.{v, u} C
        D : Type u'
        inst✝² : CategoryTheory.Category.{v', u'} D
        G : CategoryTheory.Functor C D
        inst✝¹ : CategoryTheory.Limits.HasBinaryCoproducts C
        inst✝ : CategoryTheory.Limits.HasPushouts C
        F : CategoryTheory.Functor CategoryTheory.Limits.WalkingParallelPair C
        c : CategoryTheory.Limits.Cocone F
        m : Quiver.Hom (CategoryTheory.Limits.HasCoequalizersOfHasPushoutsAndBinaryCop …
        J : ∀ (j : CategoryTheory.Limits.WalkingParallelPair), Eq (CategoryTheory.Cate …
        J1 : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.HasCoequali …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pushout.inl (C …
      -/
    · rw [colimit.ι_desc]
      /-
        case h₀
        C : Type u
        inst✝³ : CategoryTheory.Category.{v, u} C
        D : Type u'
        inst✝² : CategoryTheory.Category.{v', u'} D
        G : CategoryTheory.Functor C D
        inst✝¹ : CategoryTheory.Limits.HasBinaryCoproducts C
        inst✝ : CategoryTheory.Limits.HasPushouts C
        F : CategoryTheory.Functor CategoryTheory.Limits.WalkingParallelPair C
        c : CategoryTheory.Limits.Cocone F
        m : Quiver.Hom (CategoryTheory.Limits.HasCoequalizersOfHasPushoutsAndBinaryCop …
        J : ∀ (j : CategoryTheory.Limits.WalkingParallelPair), Eq (CategoryTheory.Cate …
        J1 : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.HasCoequali …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pushout.inl (C …
      -/
      exact J1
      /-
        🎉 no goals
      -/
      /-
        case h₁
        C : Type u
        inst✝³ : CategoryTheory.Category.{v, u} C
        D : Type u'
        inst✝² : CategoryTheory.Category.{v', u'} D
        G : CategoryTheory.Functor C D
        inst✝¹ : CategoryTheory.Limits.HasBinaryCoproducts C
        inst✝ : CategoryTheory.Limits.HasPushouts C
        F : CategoryTheory.Functor CategoryTheory.Limits.WalkingParallelPair C
        c : CategoryTheory.Limits.Cocone F
        m : Quiver.Hom (CategoryTheory.Limits.HasCoequalizersOfHasPushoutsAndBinaryCop …
        J : ∀ (j : CategoryTheory.Limits.WalkingParallelPair), Eq (CategoryTheory.Cate …
        J1 : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.HasCoequali …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pushout.inr (C …
      -/
    · rw [colimit.ι_desc, ← pushoutInl_eq_pushout_inr]
      /-
        case h₁
        C : Type u
        inst✝³ : CategoryTheory.Category.{v, u} C
        D : Type u'
        inst✝² : CategoryTheory.Category.{v', u'} D
        G : CategoryTheory.Functor C D
        inst✝¹ : CategoryTheory.Limits.HasBinaryCoproducts C
        inst✝ : CategoryTheory.Limits.HasPushouts C
        F : CategoryTheory.Functor CategoryTheory.Limits.WalkingParallelPair C
        c : CategoryTheory.Limits.Cocone F
        m : Quiver.Hom (CategoryTheory.Limits.HasCoequalizersOfHasPushoutsAndBinaryCop …
        J : ∀ (j : CategoryTheory.Limits.WalkingParallelPair), Eq (CategoryTheory.Cate …
        J1 : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.HasCoequali …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.HasCoequalizer …
      -/
      exact J1
      /-
        🎉 no goals
      -/


/-- Any category with pullbacks and binary products, has equalizers. -/
theorem hasCoequalizers_of_hasPushouts_and_binary_coproducts [HasBinaryCoproducts C]
    [HasPushouts C] : HasCoequalizers C :=
  {
    has_colimit := fun F =>
      HasColimit.mk
        { cocone := coequalizerCocone F
          isColimit := coequalizerCoconeIsColimit F } }


/-- A functor that preserves pushouts and binary coproducts also presrves coequalizers. -/
lemma preservesCoequalizers_of_preservesPushouts_and_binaryCoproducts [HasBinaryCoproducts C]
    [HasPushouts C] [PreservesColimitsOfShape (Discrete WalkingPair) G]
    [PreservesColimitsOfShape WalkingSpan G] : PreservesColimitsOfShape WalkingParallelPair G :=
  ⟨fun {K} =>
    preservesColimit_of_preserves_colimit_cocone (coequalizerCoconeIsColimit K) <|
      { desc := fun c => by
          /-
            C : Type u
            inst✝⁵ : CategoryTheory.Category.{v, u} C
            D : Type u'
            inst✝⁴ : CategoryTheory.Category.{v', u'} D
            G : CategoryTheory.Functor C D
            inst✝³ : CategoryTheory.Limits.HasBinaryCoproducts C
            inst✝² : CategoryTheory.Limits.HasPushouts C
            inst✝¹ : CategoryTheory.Limits.PreservesColimitsOfShape (CategoryTheory.Discre …
            inst✝ : CategoryTheory.Limits.PreservesColimitsOfShape CategoryTheory.Limits.W …
            K : CategoryTheory.Functor CategoryTheory.Limits.WalkingParallelPair C
            c : CategoryTheory.Limits.Cocone (K.comp G)
            ⊢ Quiver.Hom (G.mapCocone (CategoryTheory.Limits.HasCoequalizersOfHasPushoutsA …
          -/
          refine (PreservesPushout.iso _ _ _).inv ≫ pushout.desc ?_ ?_ ?_
            /-
              case refine_1
              C : Type u
              inst✝⁵ : CategoryTheory.Category.{v, u} C
              D : Type u'
              inst✝⁴ : CategoryTheory.Category.{v', u'} D
              G : CategoryTheory.Functor C D
              inst✝³ : CategoryTheory.Limits.HasBinaryCoproducts C
              inst✝² : CategoryTheory.Limits.HasPushouts C
              inst✝¹ : CategoryTheory.Limits.PreservesColimitsOfShape (CategoryTheory.Discre …
              inst✝ : CategoryTheory.Limits.PreservesColimitsOfShape CategoryTheory.Limits.W …
              K : CategoryTheory.Functor CategoryTheory.Limits.WalkingParallelPair C
              c : CategoryTheory.Limits.Cocone (K.comp G)
              ⊢ Quiver.Hom (G.obj (K.obj CategoryTheory.Limits.WalkingParallelPair.one)) c.pt
            -/
          · exact c.ι.app WalkingParallelPair.one
            /-
              🎉 no goals
            -/
            /-
              case refine_2
              C : Type u
              inst✝⁵ : CategoryTheory.Category.{v, u} C
              D : Type u'
              inst✝⁴ : CategoryTheory.Category.{v', u'} D
              G : CategoryTheory.Functor C D
              inst✝³ : CategoryTheory.Limits.HasBinaryCoproducts C
              inst✝² : CategoryTheory.Limits.HasPushouts C
              inst✝¹ : CategoryTheory.Limits.PreservesColimitsOfShape (CategoryTheory.Discre …
              inst✝ : CategoryTheory.Limits.PreservesColimitsOfShape CategoryTheory.Limits.W …
              K : CategoryTheory.Functor CategoryTheory.Limits.WalkingParallelPair C
              c : CategoryTheory.Limits.Cocone (K.comp G)
              ⊢ Quiver.Hom (G.obj (K.obj CategoryTheory.Limits.WalkingParallelPair.one)) c.pt
            -/
          · exact c.ι.app WalkingParallelPair.one
            /-
              🎉 no goals
            -/
          /-
            case refine_3
            C : Type u
            inst✝⁵ : CategoryTheory.Category.{v, u} C
            D : Type u'
            inst✝⁴ : CategoryTheory.Category.{v', u'} D
            G : CategoryTheory.Functor C D
            inst✝³ : CategoryTheory.Limits.HasBinaryCoproducts C
            inst✝² : CategoryTheory.Limits.HasPushouts C
            inst✝¹ : CategoryTheory.Limits.PreservesColimitsOfShape (CategoryTheory.Discre …
            inst✝ : CategoryTheory.Limits.PreservesColimitsOfShape CategoryTheory.Limits.W …
            K : CategoryTheory.Functor CategoryTheory.Limits.WalkingParallelPair C
            c : CategoryTheory.Limits.Cocone (K.comp G)
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.map (CategoryTheory.Limits.coprod. …
          -/
          apply (mapIsColimitOfPreservesOfIsColimit G _ _ (coprodIsCoprod _ _)).hom_ext
          /-
            case refine_3
            C : Type u
            inst✝⁵ : CategoryTheory.Category.{v, u} C
            D : Type u'
            inst✝⁴ : CategoryTheory.Category.{v', u'} D
            G : CategoryTheory.Functor C D
            inst✝³ : CategoryTheory.Limits.HasBinaryCoproducts C
            inst✝² : CategoryTheory.Limits.HasPushouts C
            inst✝¹ : CategoryTheory.Limits.PreservesColimitsOfShape (CategoryTheory.Discre …
            inst✝ : CategoryTheory.Limits.PreservesColimitsOfShape CategoryTheory.Limits.W …
            K : CategoryTheory.Functor CategoryTheory.Limits.WalkingParallelPair C
            c : CategoryTheory.Limits.Cocone (K.comp G)
            ⊢ ∀ (j : CategoryTheory.Discrete CategoryTheory.Limits.WalkingPair), Eq (Categ …
          -/
          rintro (_ | _)
          · simp only [BinaryCofan.ι_app_left, BinaryCofan.mk_inl, Category.assoc, ←
              G.map_comp_assoc, coprod.inl_desc]
          · simp only [BinaryCofan.ι_app_right, BinaryCofan.mk_inr, Category.assoc, ←
              G.map_comp_assoc, coprod.inr_desc]
            exact
              (c.ι.naturality WalkingParallelPairHom.left).trans
                (c.ι.naturality WalkingParallelPairHom.right).symm
        fac := fun c j => by
          /-
            C : Type u
            inst✝⁵ : CategoryTheory.Category.{v, u} C
            D : Type u'
            inst✝⁴ : CategoryTheory.Category.{v', u'} D
            G : CategoryTheory.Functor C D
            inst✝³ : CategoryTheory.Limits.HasBinaryCoproducts C
            inst✝² : CategoryTheory.Limits.HasPushouts C
            inst✝¹ : CategoryTheory.Limits.PreservesColimitsOfShape (CategoryTheory.Discre …
            inst✝ : CategoryTheory.Limits.PreservesColimitsOfShape CategoryTheory.Limits.W …
            K : CategoryTheory.Functor CategoryTheory.Limits.WalkingParallelPair C
            c : CategoryTheory.Limits.Cocone (K.comp G)
            j : CategoryTheory.Limits.WalkingParallelPair
            ⊢ Eq (CategoryTheory.CategoryStruct.comp ((G.mapCocone (CategoryTheory.Limits. …
          -/
          rcases j with (_ | _) <;>
            simp only [Functor.mapCocone_ι_app, Cocone.ofCofork_ι, Category.id_comp,
              eqToHom_refl, Category.assoc, Functor.map_comp, Cofork.ofπ_ι_app, pushout.inl_desc,
              PreservesPushout.inl_iso_inv_assoc]
          /-
            case zero
            C : Type u
            inst✝⁵ : CategoryTheory.Category.{v, u} C
            D : Type u'
            inst✝⁴ : CategoryTheory.Category.{v', u'} D
            G : CategoryTheory.Functor C D
            inst✝³ : CategoryTheory.Limits.HasBinaryCoproducts C
            inst✝² : CategoryTheory.Limits.HasPushouts C
            inst✝¹ : CategoryTheory.Limits.PreservesColimitsOfShape (CategoryTheory.Discre …
            inst✝ : CategoryTheory.Limits.PreservesColimitsOfShape CategoryTheory.Limits.W …
            K : CategoryTheory.Functor CategoryTheory.Limits.WalkingParallelPair C
            c : CategoryTheory.Limits.Cocone (K.comp G)
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.map (K.map CategoryTheory.Limits.W …
          -/
          exact (c.ι.naturality WalkingParallelPairHom.left).trans (Category.comp_id _)
          /-
            🎉 no goals
          -/
        uniq := fun s m h => by
          /-
            C : Type u
            inst✝⁵ : CategoryTheory.Category.{v, u} C
            D : Type u'
            inst✝⁴ : CategoryTheory.Category.{v', u'} D
            G : CategoryTheory.Functor C D
            inst✝³ : CategoryTheory.Limits.HasBinaryCoproducts C
            inst✝² : CategoryTheory.Limits.HasPushouts C
            inst✝¹ : CategoryTheory.Limits.PreservesColimitsOfShape (CategoryTheory.Discre …
            inst✝ : CategoryTheory.Limits.PreservesColimitsOfShape CategoryTheory.Limits.W …
            K : CategoryTheory.Functor CategoryTheory.Limits.WalkingParallelPair C
            s : CategoryTheory.Limits.Cocone (K.comp G)
            m : Quiver.Hom (G.mapCocone (CategoryTheory.Limits.HasCoequalizersOfHasPushout …
            h : ∀ (j : CategoryTheory.Limits.WalkingParallelPair), Eq (CategoryTheory.Cate …
            ⊢ Eq m ((fun c => CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Pr …
          -/
          rw [Iso.eq_inv_comp]
          /-
            C : Type u
            inst✝⁵ : CategoryTheory.Category.{v, u} C
            D : Type u'
            inst✝⁴ : CategoryTheory.Category.{v', u'} D
            G : CategoryTheory.Functor C D
            inst✝³ : CategoryTheory.Limits.HasBinaryCoproducts C
            inst✝² : CategoryTheory.Limits.HasPushouts C
            inst✝¹ : CategoryTheory.Limits.PreservesColimitsOfShape (CategoryTheory.Discre …
            inst✝ : CategoryTheory.Limits.PreservesColimitsOfShape CategoryTheory.Limits.W …
            K : CategoryTheory.Functor CategoryTheory.Limits.WalkingParallelPair C
            s : CategoryTheory.Limits.Cocone (K.comp G)
            m : Quiver.Hom (G.mapCocone (CategoryTheory.Limits.HasCoequalizersOfHasPushout …
            h : ∀ (j : CategoryTheory.Limits.WalkingParallelPair), Eq (CategoryTheory.Cate …
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.PreservesPusho …
          -/
          have := h WalkingParallelPair.one
          /-
            C : Type u
            inst✝⁵ : CategoryTheory.Category.{v, u} C
            D : Type u'
            inst✝⁴ : CategoryTheory.Category.{v', u'} D
            G : CategoryTheory.Functor C D
            inst✝³ : CategoryTheory.Limits.HasBinaryCoproducts C
            inst✝² : CategoryTheory.Limits.HasPushouts C
            inst✝¹ : CategoryTheory.Limits.PreservesColimitsOfShape (CategoryTheory.Discre …
            inst✝ : CategoryTheory.Limits.PreservesColimitsOfShape CategoryTheory.Limits.W …
            K : CategoryTheory.Functor CategoryTheory.Limits.WalkingParallelPair C
            s : CategoryTheory.Limits.Cocone (K.comp G)
            m : Quiver.Hom (G.mapCocone (CategoryTheory.Limits.HasCoequalizersOfHasPushout …
            h : ∀ (j : CategoryTheory.Limits.WalkingParallelPair), Eq (CategoryTheory.Cate …
            this : Eq (CategoryTheory.CategoryStruct.comp ((G.mapCocone (CategoryTheory.Li …
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.PreservesPusho …
          -/
          dsimp [coequalizerCocone] at this
          /-
            C : Type u
            inst✝⁵ : CategoryTheory.Category.{v, u} C
            D : Type u'
            inst✝⁴ : CategoryTheory.Category.{v', u'} D
            G : CategoryTheory.Functor C D
            inst✝³ : CategoryTheory.Limits.HasBinaryCoproducts C
            inst✝² : CategoryTheory.Limits.HasPushouts C
            inst✝¹ : CategoryTheory.Limits.PreservesColimitsOfShape (CategoryTheory.Discre …
            inst✝ : CategoryTheory.Limits.PreservesColimitsOfShape CategoryTheory.Limits.W …
            K : CategoryTheory.Functor CategoryTheory.Limits.WalkingParallelPair C
            s : CategoryTheory.Limits.Cocone (K.comp G)
            m : Quiver.Hom (G.mapCocone (CategoryTheory.Limits.HasCoequalizersOfHasPushout …
            h : ∀ (j : CategoryTheory.Limits.WalkingParallelPair), Eq (CategoryTheory.Cate …
            this : Eq (CategoryTheory.CategoryStruct.comp (G.map (CategoryTheory.CategoryS …
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.PreservesPusho …
          -/
          ext <;>
            simp only [PreservesPushout.inl_iso_hom_assoc, Category.id_comp, pushout.inl_desc,
              pushout.inr_desc, PreservesPushout.inr_iso_hom_assoc, ← pushoutInl_eq_pushout_inr, ←
              this] }⟩


