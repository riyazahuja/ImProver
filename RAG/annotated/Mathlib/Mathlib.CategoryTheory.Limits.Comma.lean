/-- (Implementation). An auxiliary cone which is useful in order to construct limits
in the comma category. -/
@[simps!]
def limitAuxiliaryCone (c₁ : Cone (F ⋙ fst L R)) : Cone ((F ⋙ snd L R) ⋙ R) :=
  (Cones.postcompose (whiskerLeft F (Comma.natTrans L R) : _)).obj (L.mapCone c₁)


/-- If `R` preserves the appropriate limit, then given a cone for `F ⋙ fst L R : J ⥤ L` and a
limit cone for `F ⋙ snd L R : J ⥤ R` we can build a cone for `F` which will turn out to be a limit
cone.
-/
@[simps]
noncomputable def coneOfPreserves [PreservesLimit (F ⋙ snd L R) R] (c₁ : Cone (F ⋙ fst L R))
    {c₂ : Cone (F ⋙ snd L R)} (t₂ : IsLimit c₂) : Cone F where
  pt :=
    { left := c₁.pt
      right := c₂.pt
      hom := (isLimitOfPreserves R t₂).lift (limitAuxiliaryCone _ c₁) }
  π :=
    { app := fun j =>
        { left := c₁.π.app j
          right := c₂.π.app j
          w := ((isLimitOfPreserves R t₂).fac (limitAuxiliaryCone F c₁) j).symm }
      naturality := fun j₁ j₂ t => by
        /-
          J : Type w
          inst✝⁴ : CategoryTheory.Category.{w', w} J
          A : Type u₁
          inst✝³ : CategoryTheory.Category.{v₁, u₁} A
          B : Type u₂
          inst✝² : CategoryTheory.Category.{v₂, u₂} B
          T : Type u₃
          inst✝¹ : CategoryTheory.Category.{v₃, u₃} T
          L : CategoryTheory.Functor A T
          R : CategoryTheory.Functor B T
          F : CategoryTheory.Functor J (CategoryTheory.Comma L R)
          inst✝ : CategoryTheory.Limits.PreservesLimit (F.comp (CategoryTheory.Comma.snd …
          c₁ : CategoryTheory.Limits.Cone (F.comp (CategoryTheory.Comma.fst L R))
          c₂ : CategoryTheory.Limits.Cone (F.comp (CategoryTheory.Comma.snd L R))
          t₂ : CategoryTheory.Limits.IsLimit c₂
          j₁ j₂ : J
          t : Quiver.Hom j₁ j₂
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Functor.const J).ob …
        -/
        ext
          /-
            case h₁
            J : Type w
            inst✝⁴ : CategoryTheory.Category.{w', w} J
            A : Type u₁
            inst✝³ : CategoryTheory.Category.{v₁, u₁} A
            B : Type u₂
            inst✝² : CategoryTheory.Category.{v₂, u₂} B
            T : Type u₃
            inst✝¹ : CategoryTheory.Category.{v₃, u₃} T
            L : CategoryTheory.Functor A T
            R : CategoryTheory.Functor B T
            F : CategoryTheory.Functor J (CategoryTheory.Comma L R)
            inst✝ : CategoryTheory.Limits.PreservesLimit (F.comp (CategoryTheory.Comma.snd …
            c₁ : CategoryTheory.Limits.Cone (F.comp (CategoryTheory.Comma.fst L R))
            c₂ : CategoryTheory.Limits.Cone (F.comp (CategoryTheory.Comma.snd L R))
            t₂ : CategoryTheory.Limits.IsLimit c₂
            j₁ j₂ : J
            t : Quiver.Hom j₁ j₂
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Functor.const J).ob …
          -/
        · simp [← c₁.w t]
          /-
            🎉 no goals
          -/
          /-
            case h₂
            J : Type w
            inst✝⁴ : CategoryTheory.Category.{w', w} J
            A : Type u₁
            inst✝³ : CategoryTheory.Category.{v₁, u₁} A
            B : Type u₂
            inst✝² : CategoryTheory.Category.{v₂, u₂} B
            T : Type u₃
            inst✝¹ : CategoryTheory.Category.{v₃, u₃} T
            L : CategoryTheory.Functor A T
            R : CategoryTheory.Functor B T
            F : CategoryTheory.Functor J (CategoryTheory.Comma L R)
            inst✝ : CategoryTheory.Limits.PreservesLimit (F.comp (CategoryTheory.Comma.snd …
            c₁ : CategoryTheory.Limits.Cone (F.comp (CategoryTheory.Comma.fst L R))
            c₂ : CategoryTheory.Limits.Cone (F.comp (CategoryTheory.Comma.snd L R))
            t₂ : CategoryTheory.Limits.IsLimit c₂
            j₁ j₂ : J
            t : Quiver.Hom j₁ j₂
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Functor.const J).ob …
          -/
        · simp [← c₂.w t] }
          /-
            🎉 no goals
          -/


/-- Provided that `R` preserves the appropriate limit, then the cone in `coneOfPreserves` is a
limit. -/
noncomputable def coneOfPreservesIsLimit [PreservesLimit (F ⋙ snd L R) R] {c₁ : Cone (F ⋙ fst L R)}
    (t₁ : IsLimit c₁) {c₂ : Cone (F ⋙ snd L R)} (t₂ : IsLimit c₂) :
    IsLimit (coneOfPreserves F c₁ t₂) where
  lift s :=
    { left := t₁.lift ((fst L R).mapCone s)
      right := t₂.lift ((snd L R).mapCone s)
      w :=
        (isLimitOfPreserves R t₂).hom_ext fun j => by
          rw [coneOfPreserves_pt_hom, assoc, assoc, (isLimitOfPreserves R t₂).fac,
            limitAuxiliaryCone_π_app, ← L.map_comp_assoc, t₁.fac, R.mapCone_π_app,
            ← R.map_comp, t₂.fac]
          /-
            J : Type w
            inst✝⁴ : CategoryTheory.Category.{w', w} J
            A : Type u₁
            inst✝³ : CategoryTheory.Category.{v₁, u₁} A
            B : Type u₂
            inst✝² : CategoryTheory.Category.{v₂, u₂} B
            T : Type u₃
            inst✝¹ : CategoryTheory.Category.{v₃, u₃} T
            L : CategoryTheory.Functor A T
            R : CategoryTheory.Functor B T
            F : CategoryTheory.Functor J (CategoryTheory.Comma L R)
            inst✝ : CategoryTheory.Limits.PreservesLimit (F.comp (CategoryTheory.Comma.snd …
            c₁ : CategoryTheory.Limits.Cone (F.comp (CategoryTheory.Comma.fst L R))
            t₁ : CategoryTheory.Limits.IsLimit c₁
            c₂ : CategoryTheory.Limits.Cone (F.comp (CategoryTheory.Comma.snd L R))
            t₂ : CategoryTheory.Limits.IsLimit c₂
            s : CategoryTheory.Limits.Cone F
            j : J
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (L.map (((CategoryTheory.Comma.fst L  …
          -/
          exact (s.π.app j).w }
          /-
            🎉 no goals
          -/
  uniq s m w := by
    /-
      J : Type w
      inst✝⁴ : CategoryTheory.Category.{w', w} J
      A : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} A
      B : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} B
      T : Type u₃
      inst✝¹ : CategoryTheory.Category.{v₃, u₃} T
      L : CategoryTheory.Functor A T
      R : CategoryTheory.Functor B T
      F : CategoryTheory.Functor J (CategoryTheory.Comma L R)
      inst✝ : CategoryTheory.Limits.PreservesLimit (F.comp (CategoryTheory.Comma.snd …
      c₁ : CategoryTheory.Limits.Cone (F.comp (CategoryTheory.Comma.fst L R))
      t₁ : CategoryTheory.Limits.IsLimit c₁
      c₂ : CategoryTheory.Limits.Cone (F.comp (CategoryTheory.Comma.snd L R))
      t₂ : CategoryTheory.Limits.IsLimit c₂
      s : CategoryTheory.Limits.Cone F
      m : Quiver.Hom s.pt (CategoryTheory.Comma.coneOfPreserves F c₁ t₂).pt
      w : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp m ((CategoryTheory.Comma …
      ⊢ Eq m ((fun s => { left := t₁.lift ((CategoryTheory.Comma.fst L R).mapCone s) …
    -/
    apply CommaMorphism.ext
      /-
        case left
        J : Type w
        inst✝⁴ : CategoryTheory.Category.{w', w} J
        A : Type u₁
        inst✝³ : CategoryTheory.Category.{v₁, u₁} A
        B : Type u₂
        inst✝² : CategoryTheory.Category.{v₂, u₂} B
        T : Type u₃
        inst✝¹ : CategoryTheory.Category.{v₃, u₃} T
        L : CategoryTheory.Functor A T
        R : CategoryTheory.Functor B T
        F : CategoryTheory.Functor J (CategoryTheory.Comma L R)
        inst✝ : CategoryTheory.Limits.PreservesLimit (F.comp (CategoryTheory.Comma.snd …
        c₁ : CategoryTheory.Limits.Cone (F.comp (CategoryTheory.Comma.fst L R))
        t₁ : CategoryTheory.Limits.IsLimit c₁
        c₂ : CategoryTheory.Limits.Cone (F.comp (CategoryTheory.Comma.snd L R))
        t₂ : CategoryTheory.Limits.IsLimit c₂
        s : CategoryTheory.Limits.Cone F
        m : Quiver.Hom s.pt (CategoryTheory.Comma.coneOfPreserves F c₁ t₂).pt
        w : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp m ((CategoryTheory.Comma …
        ⊢ Eq m.left ((fun s => { left := t₁.lift ((CategoryTheory.Comma.fst L R).mapCo …
      -/
    · exact t₁.uniq ((fst L R).mapCone s) _ (fun j => by simp [← w])
      /-
        🎉 no goals
      -/
      /-
        case right
        J : Type w
        inst✝⁴ : CategoryTheory.Category.{w', w} J
        A : Type u₁
        inst✝³ : CategoryTheory.Category.{v₁, u₁} A
        B : Type u₂
        inst✝² : CategoryTheory.Category.{v₂, u₂} B
        T : Type u₃
        inst✝¹ : CategoryTheory.Category.{v₃, u₃} T
        L : CategoryTheory.Functor A T
        R : CategoryTheory.Functor B T
        F : CategoryTheory.Functor J (CategoryTheory.Comma L R)
        inst✝ : CategoryTheory.Limits.PreservesLimit (F.comp (CategoryTheory.Comma.snd …
        c₁ : CategoryTheory.Limits.Cone (F.comp (CategoryTheory.Comma.fst L R))
        t₁ : CategoryTheory.Limits.IsLimit c₁
        c₂ : CategoryTheory.Limits.Cone (F.comp (CategoryTheory.Comma.snd L R))
        t₂ : CategoryTheory.Limits.IsLimit c₂
        s : CategoryTheory.Limits.Cone F
        m : Quiver.Hom s.pt (CategoryTheory.Comma.coneOfPreserves F c₁ t₂).pt
        w : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp m ((CategoryTheory.Comma …
        ⊢ Eq m.right ((fun s => { left := t₁.lift ((CategoryTheory.Comma.fst L R).mapC …
      -/
    · exact t₂.uniq ((snd L R).mapCone s) _ (fun j => by simp [← w])
      /-
        🎉 no goals
      -/


/-- (Implementation). An auxiliary cocone which is useful in order to construct colimits
in the comma category. -/
@[simps!]
def colimitAuxiliaryCocone (c₂ : Cocone (F ⋙ snd L R)) : Cocone ((F ⋙ fst L R) ⋙ L) :=
  (Cocones.precompose (whiskerLeft F (Comma.natTrans L R) : _)).obj (R.mapCocone c₂)


/--
If `L` preserves the appropriate colimit, then given a colimit cocone for `F ⋙ fst L R : J ⥤ L` and
a cocone for `F ⋙ snd L R : J ⥤ R` we can build a cocone for `F` which will turn out to be a
colimit cocone.
-/
@[simps]
noncomputable def coconeOfPreserves [PreservesColimit (F ⋙ fst L R) L] {c₁ : Cocone (F ⋙ fst L R)}
    (t₁ : IsColimit c₁) (c₂ : Cocone (F ⋙ snd L R)) : Cocone F where
  pt :=
    { left := c₁.pt
      right := c₂.pt
      hom := (isColimitOfPreserves L t₁).desc (colimitAuxiliaryCocone _ c₂) }
  ι :=
    { app := fun j =>
        { left := c₁.ι.app j
          right := c₂.ι.app j
          w := (isColimitOfPreserves L t₁).fac (colimitAuxiliaryCocone _ c₂) j }
      naturality := fun j₁ j₂ t => by
        /-
          J : Type w
          inst✝⁴ : CategoryTheory.Category.{w', w} J
          A : Type u₁
          inst✝³ : CategoryTheory.Category.{v₁, u₁} A
          B : Type u₂
          inst✝² : CategoryTheory.Category.{v₂, u₂} B
          T : Type u₃
          inst✝¹ : CategoryTheory.Category.{v₃, u₃} T
          L : CategoryTheory.Functor A T
          R : CategoryTheory.Functor B T
          F : CategoryTheory.Functor J (CategoryTheory.Comma L R)
          inst✝ : CategoryTheory.Limits.PreservesColimit (F.comp (CategoryTheory.Comma.f …
          c₁ : CategoryTheory.Limits.Cocone (F.comp (CategoryTheory.Comma.fst L R))
          t₁ : CategoryTheory.Limits.IsColimit c₁
          c₂ : CategoryTheory.Limits.Cocone (F.comp (CategoryTheory.Comma.snd L R))
          j₁ j₂ : J
          t : Quiver.Hom j₁ j₂
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map t) ((fun j => { left := c₁.ι.a …
        -/
        ext
          /-
            case h₁
            J : Type w
            inst✝⁴ : CategoryTheory.Category.{w', w} J
            A : Type u₁
            inst✝³ : CategoryTheory.Category.{v₁, u₁} A
            B : Type u₂
            inst✝² : CategoryTheory.Category.{v₂, u₂} B
            T : Type u₃
            inst✝¹ : CategoryTheory.Category.{v₃, u₃} T
            L : CategoryTheory.Functor A T
            R : CategoryTheory.Functor B T
            F : CategoryTheory.Functor J (CategoryTheory.Comma L R)
            inst✝ : CategoryTheory.Limits.PreservesColimit (F.comp (CategoryTheory.Comma.f …
            c₁ : CategoryTheory.Limits.Cocone (F.comp (CategoryTheory.Comma.fst L R))
            t₁ : CategoryTheory.Limits.IsColimit c₁
            c₂ : CategoryTheory.Limits.Cocone (F.comp (CategoryTheory.Comma.snd L R))
            j₁ j₂ : J
            t : Quiver.Hom j₁ j₂
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map t) ((fun j => { left := c₁.ι.a …
          -/
        · simp [← c₁.w t]
          /-
            🎉 no goals
          -/
          /-
            case h₂
            J : Type w
            inst✝⁴ : CategoryTheory.Category.{w', w} J
            A : Type u₁
            inst✝³ : CategoryTheory.Category.{v₁, u₁} A
            B : Type u₂
            inst✝² : CategoryTheory.Category.{v₂, u₂} B
            T : Type u₃
            inst✝¹ : CategoryTheory.Category.{v₃, u₃} T
            L : CategoryTheory.Functor A T
            R : CategoryTheory.Functor B T
            F : CategoryTheory.Functor J (CategoryTheory.Comma L R)
            inst✝ : CategoryTheory.Limits.PreservesColimit (F.comp (CategoryTheory.Comma.f …
            c₁ : CategoryTheory.Limits.Cocone (F.comp (CategoryTheory.Comma.fst L R))
            t₁ : CategoryTheory.Limits.IsColimit c₁
            c₂ : CategoryTheory.Limits.Cocone (F.comp (CategoryTheory.Comma.snd L R))
            j₁ j₂ : J
            t : Quiver.Hom j₁ j₂
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map t) ((fun j => { left := c₁.ι.a …
          -/
        · simp [← c₂.w t] }
          /-
            🎉 no goals
          -/


/-- Provided that `L` preserves the appropriate colimit, then the cocone in `coconeOfPreserves` is
a colimit. -/
noncomputable def coconeOfPreservesIsColimit [PreservesColimit (F ⋙ fst L R) L]
    {c₁ : Cocone (F ⋙ fst L R)}
    (t₁ : IsColimit c₁) {c₂ : Cocone (F ⋙ snd L R)} (t₂ : IsColimit c₂) :
    IsColimit (coconeOfPreserves F t₁ c₂) where
  desc s :=
    { left := t₁.desc ((fst L R).mapCocone s)
      right := t₂.desc ((snd L R).mapCocone s)
      w :=
        (isColimitOfPreserves L t₁).hom_ext fun j => by
          rw [coconeOfPreserves_pt_hom, (isColimitOfPreserves L t₁).fac_assoc,
            colimitAuxiliaryCocone_ι_app, assoc, ← R.map_comp, t₂.fac, L.mapCocone_ι_app, ←
            L.map_comp_assoc, t₁.fac]
          /-
            J : Type w
            inst✝⁴ : CategoryTheory.Category.{w', w} J
            A : Type u₁
            inst✝³ : CategoryTheory.Category.{v₁, u₁} A
            B : Type u₂
            inst✝² : CategoryTheory.Category.{v₂, u₂} B
            T : Type u₃
            inst✝¹ : CategoryTheory.Category.{v₃, u₃} T
            L : CategoryTheory.Functor A T
            R : CategoryTheory.Functor B T
            F : CategoryTheory.Functor J (CategoryTheory.Comma L R)
            inst✝ : CategoryTheory.Limits.PreservesColimit (F.comp (CategoryTheory.Comma.f …
            c₁ : CategoryTheory.Limits.Cocone (F.comp (CategoryTheory.Comma.fst L R))
            t₁ : CategoryTheory.Limits.IsColimit c₁
            c₂ : CategoryTheory.Limits.Cocone (F.comp (CategoryTheory.Comma.snd L R))
            t₂ : CategoryTheory.Limits.IsColimit c₂
            s : CategoryTheory.Limits.Cocone F
            j : J
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (L.map (((CategoryTheory.Comma.fst L  …
          -/
          exact (s.ι.app j).w }
          /-
            🎉 no goals
          -/
  uniq s m w := by
    /-
      J : Type w
      inst✝⁴ : CategoryTheory.Category.{w', w} J
      A : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} A
      B : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} B
      T : Type u₃
      inst✝¹ : CategoryTheory.Category.{v₃, u₃} T
      L : CategoryTheory.Functor A T
      R : CategoryTheory.Functor B T
      F : CategoryTheory.Functor J (CategoryTheory.Comma L R)
      inst✝ : CategoryTheory.Limits.PreservesColimit (F.comp (CategoryTheory.Comma.f …
      c₁ : CategoryTheory.Limits.Cocone (F.comp (CategoryTheory.Comma.fst L R))
      t₁ : CategoryTheory.Limits.IsColimit c₁
      c₂ : CategoryTheory.Limits.Cocone (F.comp (CategoryTheory.Comma.snd L R))
      t₂ : CategoryTheory.Limits.IsColimit c₂
      s : CategoryTheory.Limits.Cocone F
      m : Quiver.Hom (CategoryTheory.Comma.coconeOfPreserves F t₁ c₂).pt s.pt
      w : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Comma.c …
      ⊢ Eq m ((fun s => { left := t₁.desc ((CategoryTheory.Comma.fst L R).mapCocone  …
    -/
    apply CommaMorphism.ext
      /-
        case left
        J : Type w
        inst✝⁴ : CategoryTheory.Category.{w', w} J
        A : Type u₁
        inst✝³ : CategoryTheory.Category.{v₁, u₁} A
        B : Type u₂
        inst✝² : CategoryTheory.Category.{v₂, u₂} B
        T : Type u₃
        inst✝¹ : CategoryTheory.Category.{v₃, u₃} T
        L : CategoryTheory.Functor A T
        R : CategoryTheory.Functor B T
        F : CategoryTheory.Functor J (CategoryTheory.Comma L R)
        inst✝ : CategoryTheory.Limits.PreservesColimit (F.comp (CategoryTheory.Comma.f …
        c₁ : CategoryTheory.Limits.Cocone (F.comp (CategoryTheory.Comma.fst L R))
        t₁ : CategoryTheory.Limits.IsColimit c₁
        c₂ : CategoryTheory.Limits.Cocone (F.comp (CategoryTheory.Comma.snd L R))
        t₂ : CategoryTheory.Limits.IsColimit c₂
        s : CategoryTheory.Limits.Cocone F
        m : Quiver.Hom (CategoryTheory.Comma.coconeOfPreserves F t₁ c₂).pt s.pt
        w : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Comma.c …
        ⊢ Eq m.left ((fun s => { left := t₁.desc ((CategoryTheory.Comma.fst L R).mapCo …
      -/
    · exact t₁.uniq ((fst L R).mapCocone s) _ (fun j => by simp [← w])
      /-
        🎉 no goals
      -/
      /-
        case right
        J : Type w
        inst✝⁴ : CategoryTheory.Category.{w', w} J
        A : Type u₁
        inst✝³ : CategoryTheory.Category.{v₁, u₁} A
        B : Type u₂
        inst✝² : CategoryTheory.Category.{v₂, u₂} B
        T : Type u₃
        inst✝¹ : CategoryTheory.Category.{v₃, u₃} T
        L : CategoryTheory.Functor A T
        R : CategoryTheory.Functor B T
        F : CategoryTheory.Functor J (CategoryTheory.Comma L R)
        inst✝ : CategoryTheory.Limits.PreservesColimit (F.comp (CategoryTheory.Comma.f …
        c₁ : CategoryTheory.Limits.Cocone (F.comp (CategoryTheory.Comma.fst L R))
        t₁ : CategoryTheory.Limits.IsColimit c₁
        c₂ : CategoryTheory.Limits.Cocone (F.comp (CategoryTheory.Comma.snd L R))
        t₂ : CategoryTheory.Limits.IsColimit c₂
        s : CategoryTheory.Limits.Cocone F
        m : Quiver.Hom (CategoryTheory.Comma.coconeOfPreserves F t₁ c₂).pt s.pt
        w : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Comma.c …
        ⊢ Eq m.right ((fun s => { left := t₁.desc ((CategoryTheory.Comma.fst L R).mapC …
      -/
    · exact t₂.uniq ((snd L R).mapCocone s) _ (fun j => by simp [← w])
      /-
        🎉 no goals
      -/


instance hasLimit (F : J ⥤ Comma L R) [HasLimit (F ⋙ fst L R)] [HasLimit (F ⋙ snd L R)]
    [PreservesLimit (F ⋙ snd L R) R] : HasLimit F :=
  HasLimit.mk ⟨_, coneOfPreservesIsLimit _ (limit.isLimit _) (limit.isLimit _)⟩


instance hasLimitsOfShape [HasLimitsOfShape J A] [HasLimitsOfShape J B]
    [PreservesLimitsOfShape J R] : HasLimitsOfShape J (Comma L R) where


instance hasLimitsOfSize [HasLimitsOfSize.{w, w'} A] [HasLimitsOfSize.{w, w'} B]
    [PreservesLimitsOfSize.{w, w'} R] : HasLimitsOfSize.{w, w'} (Comma L R) :=
  ⟨fun _ _ => inferInstance⟩


instance hasColimit (F : J ⥤ Comma L R) [HasColimit (F ⋙ fst L R)] [HasColimit (F ⋙ snd L R)]
    [PreservesColimit (F ⋙ fst L R) L] : HasColimit F :=
  HasColimit.mk ⟨_, coconeOfPreservesIsColimit _ (colimit.isColimit _) (colimit.isColimit _)⟩


instance hasColimitsOfShape [HasColimitsOfShape J A] [HasColimitsOfShape J B]
    [PreservesColimitsOfShape J L] : HasColimitsOfShape J (Comma L R) where


instance hasColimitsOfSize [HasColimitsOfSize.{w, w'} A] [HasColimitsOfSize.{w, w'} B]
    [PreservesColimitsOfSize.{w, w'} L] : HasColimitsOfSize.{w, w'} (Comma L R) :=
  ⟨fun _ _ => inferInstance⟩


instance hasLimit (F : J ⥤ Arrow T) [i₁ : HasLimit (F ⋙ leftFunc)] [i₂ : HasLimit (F ⋙ rightFunc)] :
    HasLimit F := by
  /-
    J : Type w
    inst✝³ : CategoryTheory.Category.{w', w} J
    A : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} A
    B : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} B
    T : Type u₃
    inst✝ : CategoryTheory.Category.{v₃, u₃} T
    F : CategoryTheory.Functor J (CategoryTheory.Arrow T)
    i₁ : CategoryTheory.Limits.HasLimit (F.comp CategoryTheory.Arrow.leftFunc)
    i₂ : CategoryTheory.Limits.HasLimit (F.comp CategoryTheory.Arrow.rightFunc)
    ⊢ CategoryTheory.Limits.HasLimit F
  -/
  haveI : HasLimit (F ⋙ Comma.fst _ _) := i₁
  /-
    J : Type w
    inst✝³ : CategoryTheory.Category.{w', w} J
    A : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} A
    B : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} B
    T : Type u₃
    inst✝ : CategoryTheory.Category.{v₃, u₃} T
    F : CategoryTheory.Functor J (CategoryTheory.Arrow T)
    i₁ : CategoryTheory.Limits.HasLimit (F.comp CategoryTheory.Arrow.leftFunc)
    i₂ : CategoryTheory.Limits.HasLimit (F.comp CategoryTheory.Arrow.rightFunc)
    this : CategoryTheory.Limits.HasLimit (F.comp (CategoryTheory.Comma.fst (Categ …
    ⊢ CategoryTheory.Limits.HasLimit F
  -/
  haveI : HasLimit (F ⋙ Comma.snd _ _) := i₂
  /-
    J : Type w
    inst✝³ : CategoryTheory.Category.{w', w} J
    A : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} A
    B : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} B
    T : Type u₃
    inst✝ : CategoryTheory.Category.{v₃, u₃} T
    F : CategoryTheory.Functor J (CategoryTheory.Arrow T)
    i₁ : CategoryTheory.Limits.HasLimit (F.comp CategoryTheory.Arrow.leftFunc)
    i₂ : CategoryTheory.Limits.HasLimit (F.comp CategoryTheory.Arrow.rightFunc)
    this✝ : CategoryTheory.Limits.HasLimit (F.comp (CategoryTheory.Comma.fst (Cate …
    this : CategoryTheory.Limits.HasLimit (F.comp (CategoryTheory.Comma.snd (Categ …
    ⊢ CategoryTheory.Limits.HasLimit F
  -/
  apply Comma.hasLimit
  /-
    🎉 no goals
  -/


instance hasLimitsOfShape [HasLimitsOfShape J T] : HasLimitsOfShape J (Arrow T) where


instance hasLimits [HasLimits T] : HasLimits (Arrow T) :=
  ⟨fun _ _ => inferInstance⟩


instance hasColimit (F : J ⥤ Arrow T) [i₁ : HasColimit (F ⋙ leftFunc)]
    [i₂ : HasColimit (F ⋙ rightFunc)] : HasColimit F := by
  /-
    J : Type w
    inst✝³ : CategoryTheory.Category.{w', w} J
    A : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} A
    B : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} B
    T : Type u₃
    inst✝ : CategoryTheory.Category.{v₃, u₃} T
    F : CategoryTheory.Functor J (CategoryTheory.Arrow T)
    i₁ : CategoryTheory.Limits.HasColimit (F.comp CategoryTheory.Arrow.leftFunc)
    i₂ : CategoryTheory.Limits.HasColimit (F.comp CategoryTheory.Arrow.rightFunc)
    ⊢ CategoryTheory.Limits.HasColimit F
  -/
  haveI : HasColimit (F ⋙ Comma.fst _ _) := i₁
  /-
    J : Type w
    inst✝³ : CategoryTheory.Category.{w', w} J
    A : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} A
    B : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} B
    T : Type u₃
    inst✝ : CategoryTheory.Category.{v₃, u₃} T
    F : CategoryTheory.Functor J (CategoryTheory.Arrow T)
    i₁ : CategoryTheory.Limits.HasColimit (F.comp CategoryTheory.Arrow.leftFunc)
    i₂ : CategoryTheory.Limits.HasColimit (F.comp CategoryTheory.Arrow.rightFunc)
    this : CategoryTheory.Limits.HasColimit (F.comp (CategoryTheory.Comma.fst (Cat …
    ⊢ CategoryTheory.Limits.HasColimit F
  -/
  haveI : HasColimit (F ⋙ Comma.snd _ _) := i₂
  /-
    J : Type w
    inst✝³ : CategoryTheory.Category.{w', w} J
    A : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} A
    B : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} B
    T : Type u₃
    inst✝ : CategoryTheory.Category.{v₃, u₃} T
    F : CategoryTheory.Functor J (CategoryTheory.Arrow T)
    i₁ : CategoryTheory.Limits.HasColimit (F.comp CategoryTheory.Arrow.leftFunc)
    i₂ : CategoryTheory.Limits.HasColimit (F.comp CategoryTheory.Arrow.rightFunc)
    this✝ : CategoryTheory.Limits.HasColimit (F.comp (CategoryTheory.Comma.fst (Ca …
    this : CategoryTheory.Limits.HasColimit (F.comp (CategoryTheory.Comma.snd (Cat …
    ⊢ CategoryTheory.Limits.HasColimit F
  -/
  apply Comma.hasColimit
  /-
    🎉 no goals
  -/


instance hasColimitsOfShape [HasColimitsOfShape J T] : HasColimitsOfShape J (Arrow T) where


instance hasColimits [HasColimits T] : HasColimits (Arrow T) :=
  ⟨fun _ _ => inferInstance⟩


instance hasLimit [i₁ : HasLimit (F ⋙ proj X G)] [i₂ : PreservesLimit (F ⋙ proj X G) G] :
    HasLimit F := by
  /-
    J : Type w
    inst✝³ : CategoryTheory.Category.{w', w} J
    A : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} A
    B : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} B
    T : Type u₃
    inst✝ : CategoryTheory.Category.{v₃, u₃} T
    X : T
    G : CategoryTheory.Functor A T
    F : CategoryTheory.Functor J (CategoryTheory.StructuredArrow X G)
    i₁ : CategoryTheory.Limits.HasLimit (F.comp (CategoryTheory.StructuredArrow.pr …
    i₂ : CategoryTheory.Limits.PreservesLimit (F.comp (CategoryTheory.StructuredAr …
    ⊢ CategoryTheory.Limits.HasLimit F
  -/
  haveI : HasLimit (F ⋙ Comma.snd (Functor.fromPUnit X) G) := i₁
  /-
    J : Type w
    inst✝³ : CategoryTheory.Category.{w', w} J
    A : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} A
    B : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} B
    T : Type u₃
    inst✝ : CategoryTheory.Category.{v₃, u₃} T
    X : T
    G : CategoryTheory.Functor A T
    F : CategoryTheory.Functor J (CategoryTheory.StructuredArrow X G)
    i₁ : CategoryTheory.Limits.HasLimit (F.comp (CategoryTheory.StructuredArrow.pr …
    i₂ : CategoryTheory.Limits.PreservesLimit (F.comp (CategoryTheory.StructuredAr …
    this : CategoryTheory.Limits.HasLimit (F.comp (CategoryTheory.Comma.snd (Categ …
    ⊢ CategoryTheory.Limits.HasLimit F
  -/
  haveI : PreservesLimit (F ⋙ Comma.snd (Functor.fromPUnit X) G) _ := i₂
  /-
    J : Type w
    inst✝³ : CategoryTheory.Category.{w', w} J
    A : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} A
    B : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} B
    T : Type u₃
    inst✝ : CategoryTheory.Category.{v₃, u₃} T
    X : T
    G : CategoryTheory.Functor A T
    F : CategoryTheory.Functor J (CategoryTheory.StructuredArrow X G)
    i₁ : CategoryTheory.Limits.HasLimit (F.comp (CategoryTheory.StructuredArrow.pr …
    i₂ : CategoryTheory.Limits.PreservesLimit (F.comp (CategoryTheory.StructuredAr …
    this✝ : CategoryTheory.Limits.HasLimit (F.comp (CategoryTheory.Comma.snd (Cate …
    this : CategoryTheory.Limits.PreservesLimit (F.comp (CategoryTheory.Comma.snd  …
    ⊢ CategoryTheory.Limits.HasLimit F
  -/
  apply Comma.hasLimit
  /-
    🎉 no goals
  -/


instance hasLimitsOfShape [HasLimitsOfShape J A] [PreservesLimitsOfShape J G] :
    HasLimitsOfShape J (StructuredArrow X G) where


instance hasLimitsOfSize [HasLimitsOfSize.{w, w'} A] [PreservesLimitsOfSize.{w, w'} G] :
    HasLimitsOfSize.{w, w'} (StructuredArrow X G) :=
                  /-
                    J✝ : Type w
                    inst✝⁵ : CategoryTheory.Category.{w', w} J✝
                    A : Type u₁
                    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} A
                    B : Type u₂
                    inst✝³ : CategoryTheory.Category.{v₂, u₂} B
                    T : Type u₃
                    inst✝² : CategoryTheory.Category.{v₃, u₃} T
                    X : T
                    G : CategoryTheory.Functor A T
                    F : CategoryTheory.Functor J✝ (CategoryTheory.StructuredArrow X G)
                    inst✝¹ : CategoryTheory.Limits.HasLimitsOfSize.{w, w', v₁, u₁} A
                    inst✝ : CategoryTheory.Limits.PreservesLimitsOfSize.{w, w', v₁, v₃, u₁, u₃} G
                    J : Type w'
                    hJ : CategoryTheory.Category.{w, w'} J
                    ⊢ CategoryTheory.Limits.HasLimitsOfShape J (CategoryTheory.StructuredArrow X G)
                  -/
  ⟨fun J hJ => by infer_instance⟩
                  /-
                    🎉 no goals
                  -/


noncomputable instance createsLimit [i : PreservesLimit (F ⋙ proj X G) G] :
    CreatesLimit F (proj X G) :=
  letI : PreservesLimit (F ⋙ Comma.snd (Functor.fromPUnit X) G) G := i
  createsLimitOfReflectsIso fun _ t =>
    { liftedCone := Comma.coneOfPreserves F punitCone t
      makesLimit := Comma.coneOfPreservesIsLimit _ punitConeIsLimit _
      validLift := Cones.ext (Iso.refl _) fun _ => (id_comp _).symm }


noncomputable instance createsLimitsOfShape [PreservesLimitsOfShape J G] :
    CreatesLimitsOfShape J (proj X G) where


noncomputable instance createsLimitsOfSize [PreservesLimitsOfSize.{w, w'} G] :
    CreatesLimitsOfSize.{w, w'} (proj X G : _) where


instance mono_right_of_mono [HasPullbacks A] [PreservesLimitsOfShape WalkingCospan G]
    {Y Z : StructuredArrow X G} (f : Y ⟶ Z) [Mono f] : Mono f.right :=
  show Mono ((proj X G).map f) from inferInstance


theorem mono_iff_mono_right [HasPullbacks A] [PreservesLimitsOfShape WalkingCospan G]
    {Y Z : StructuredArrow X G} (f : Y ⟶ Z) : Mono f ↔ Mono f.right :=
  ⟨fun _ => inferInstance, fun _ => mono_of_mono_right f⟩


instance hasTerminal [G.Faithful] [G.Full] {Y : A} :
    HasTerminal (CostructuredArrow G (G.obj Y)) :=
  CostructuredArrow.mkIdTerminal.hasTerminal


instance hasColimit [i₁ : HasColimit (F ⋙ proj G X)] [i₂ : PreservesColimit (F ⋙ proj G X) G] :
    HasColimit F := by
  /-
    J : Type w
    inst✝³ : CategoryTheory.Category.{w', w} J
    A : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} A
    B : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} B
    T : Type u₃
    inst✝ : CategoryTheory.Category.{v₃, u₃} T
    G : CategoryTheory.Functor A T
    X : T
    F : CategoryTheory.Functor J (CategoryTheory.CostructuredArrow G X)
    i₁ : CategoryTheory.Limits.HasColimit (F.comp (CategoryTheory.CostructuredArro …
    i₂ : CategoryTheory.Limits.PreservesColimit (F.comp (CategoryTheory.Costructur …
    ⊢ CategoryTheory.Limits.HasColimit F
  -/
  haveI : HasColimit (F ⋙ Comma.fst G (Functor.fromPUnit X)) := i₁
  /-
    J : Type w
    inst✝³ : CategoryTheory.Category.{w', w} J
    A : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} A
    B : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} B
    T : Type u₃
    inst✝ : CategoryTheory.Category.{v₃, u₃} T
    G : CategoryTheory.Functor A T
    X : T
    F : CategoryTheory.Functor J (CategoryTheory.CostructuredArrow G X)
    i₁ : CategoryTheory.Limits.HasColimit (F.comp (CategoryTheory.CostructuredArro …
    i₂ : CategoryTheory.Limits.PreservesColimit (F.comp (CategoryTheory.Costructur …
    this : CategoryTheory.Limits.HasColimit (F.comp (CategoryTheory.Comma.fst G (C …
    ⊢ CategoryTheory.Limits.HasColimit F
  -/
  haveI : PreservesColimit (F ⋙ Comma.fst G (Functor.fromPUnit X)) _ := i₂
  /-
    J : Type w
    inst✝³ : CategoryTheory.Category.{w', w} J
    A : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} A
    B : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} B
    T : Type u₃
    inst✝ : CategoryTheory.Category.{v₃, u₃} T
    G : CategoryTheory.Functor A T
    X : T
    F : CategoryTheory.Functor J (CategoryTheory.CostructuredArrow G X)
    i₁ : CategoryTheory.Limits.HasColimit (F.comp (CategoryTheory.CostructuredArro …
    i₂ : CategoryTheory.Limits.PreservesColimit (F.comp (CategoryTheory.Costructur …
    this✝ : CategoryTheory.Limits.HasColimit (F.comp (CategoryTheory.Comma.fst G ( …
    this : CategoryTheory.Limits.PreservesColimit (F.comp (CategoryTheory.Comma.fs …
    ⊢ CategoryTheory.Limits.HasColimit F
  -/
  apply Comma.hasColimit
  /-
    🎉 no goals
  -/


instance hasColimitsOfShape [HasColimitsOfShape J A] [PreservesColimitsOfShape J G] :
    HasColimitsOfShape J (CostructuredArrow G X) where


instance hasColimitsOfSize [HasColimitsOfSize.{w, w'} A] [PreservesColimitsOfSize.{w, w'} G] :
    HasColimitsOfSize.{w, w'} (CostructuredArrow G X) :=
  ⟨fun _ _ => inferInstance⟩


noncomputable instance createsColimit [i : PreservesColimit (F ⋙ proj G X) G] :
    CreatesColimit F (proj G X) :=
  letI : PreservesColimit (F ⋙ Comma.fst G (Functor.fromPUnit X)) G := i
  createsColimitOfReflectsIso fun _ t =>
    { liftedCocone := Comma.coconeOfPreserves F t punitCocone
      makesColimit := Comma.coconeOfPreservesIsColimit _ _ punitCoconeIsColimit
      validLift := Cocones.ext (Iso.refl _) fun _ => comp_id _ }


noncomputable instance createsColimitsOfShape [PreservesColimitsOfShape J G] :
    CreatesColimitsOfShape J (proj G X) where


noncomputable instance createsColimitsOfSize [PreservesColimitsOfSize.{w, w'} G] :
    CreatesColimitsOfSize.{w, w'} (proj G X : _) where


instance epi_left_of_epi [HasPushouts A] [PreservesColimitsOfShape WalkingSpan G]
    {Y Z : CostructuredArrow G X} (f : Y ⟶ Z) [Epi f] : Epi f.left :=
  show Epi ((proj G X).map f) from inferInstance


theorem epi_iff_epi_left [HasPushouts A] [PreservesColimitsOfShape WalkingSpan G]
    {Y Z : CostructuredArrow G X} (f : Y ⟶ Z) : Epi f ↔ Epi f.left :=
  ⟨fun _ => inferInstance, fun _ => epi_of_epi_left f⟩


instance {X : T} : HasTerminal (Over X) := CostructuredArrow.hasTerminal


