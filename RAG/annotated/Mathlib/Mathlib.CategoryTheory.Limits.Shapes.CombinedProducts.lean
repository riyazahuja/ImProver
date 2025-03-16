/-- For fans on maps `f₁ : ι₁ → C`, `f₂ : ι₂ → C` and a binary fan on their
cone points, construct one family of morphisms indexed by `ι₁ ⊕ ι₂` -/
@[simp]
abbrev combPairHoms : (i : ι₁ ⊕ ι₂) → bc.pt ⟶ Sum.elim f₁ f₂ i
  | .inl a => bc.fst ≫ c₁.proj a
  | .inr a => bc.snd ≫ c₂.proj a


/-- If `c₁` and `c₂` are limit fans and `bc` is a limit binary fan on their cone
points, then the fan constructed from `combPairHoms` is a limit cone. -/
def combPairIsLimit : IsLimit (Fan.mk bc.pt (combPairHoms c₁ c₂ bc)) :=
  mkFanLimit _
    (fun s ↦ Fan.IsLimit.desc h <| fun i ↦ by
      /-
        C : Type u₁
        inst✝ : CategoryTheory.Category.{u₂, u₁} C
        ι₁ : Type u_1
        ι₂ : Type u_2
        X : C
        f₁ : ι₁ → C
        f₂ : ι₂ → C
        c₁ : CategoryTheory.Limits.Fan f₁
        c₂ : CategoryTheory.Limits.Fan f₂
        bc : CategoryTheory.Limits.BinaryFan c₁.pt c₂.pt
        h₁ : CategoryTheory.Limits.IsLimit c₁
        h₂ : CategoryTheory.Limits.IsLimit c₂
        h : CategoryTheory.Limits.IsLimit bc
        s : CategoryTheory.Limits.Fan (Sum.elim f₁ f₂)
        i : CategoryTheory.Limits.WalkingPair
        ⊢ Quiver.Hom s.pt (CategoryTheory.Limits.WalkingPair.casesOn i c₁.pt c₂.pt)
      -/
      cases i
        /-
          case left
          C : Type u₁
          inst✝ : CategoryTheory.Category.{u₂, u₁} C
          ι₁ : Type u_1
          ι₂ : Type u_2
          X : C
          f₁ : ι₁ → C
          f₂ : ι₂ → C
          c₁ : CategoryTheory.Limits.Fan f₁
          c₂ : CategoryTheory.Limits.Fan f₂
          bc : CategoryTheory.Limits.BinaryFan c₁.pt c₂.pt
          h₁ : CategoryTheory.Limits.IsLimit c₁
          h₂ : CategoryTheory.Limits.IsLimit c₂
          h : CategoryTheory.Limits.IsLimit bc
          s : CategoryTheory.Limits.Fan (Sum.elim f₁ f₂)
          ⊢ Quiver.Hom s.pt (CategoryTheory.Limits.WalkingPair.casesOn CategoryTheory.Li …
        -/
      · exact Fan.IsLimit.desc h₁ (fun a ↦ s.proj (.inl a))
        /-
          🎉 no goals
        -/
        /-
          case right
          C : Type u₁
          inst✝ : CategoryTheory.Category.{u₂, u₁} C
          ι₁ : Type u_1
          ι₂ : Type u_2
          X : C
          f₁ : ι₁ → C
          f₂ : ι₂ → C
          c₁ : CategoryTheory.Limits.Fan f₁
          c₂ : CategoryTheory.Limits.Fan f₂
          bc : CategoryTheory.Limits.BinaryFan c₁.pt c₂.pt
          h₁ : CategoryTheory.Limits.IsLimit c₁
          h₂ : CategoryTheory.Limits.IsLimit c₂
          h : CategoryTheory.Limits.IsLimit bc
          s : CategoryTheory.Limits.Fan (Sum.elim f₁ f₂)
          ⊢ Quiver.Hom s.pt (CategoryTheory.Limits.WalkingPair.casesOn CategoryTheory.Li …
        -/
      · exact Fan.IsLimit.desc h₂ (fun a ↦ s.proj (.inr a)))
        /-
          🎉 no goals
        -/
    (fun s w ↦ by
      /-
        C : Type u₁
        inst✝ : CategoryTheory.Category.{u₂, u₁} C
        ι₁ : Type u_1
        ι₂ : Type u_2
        X : C
        f₁ : ι₁ → C
        f₂ : ι₂ → C
        c₁ : CategoryTheory.Limits.Fan f₁
        c₂ : CategoryTheory.Limits.Fan f₂
        bc : CategoryTheory.Limits.BinaryFan c₁.pt c₂.pt
        h₁ : CategoryTheory.Limits.IsLimit c₁
        h₂ : CategoryTheory.Limits.IsLimit c₂
        h : CategoryTheory.Limits.IsLimit bc
        s : CategoryTheory.Limits.Fan (Sum.elim f₁ f₂)
        w : Sum ι₁ ι₂
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun s => CategoryTheory.Limits.Fan. …
      -/
      cases w <;>
        /-
          case inl
          C : Type u₁
          inst✝ : CategoryTheory.Category.{u₂, u₁} C
          ι₁ : Type u_1
          ι₂ : Type u_2
          X : C
          f₁ : ι₁ → C
          f₂ : ι₂ → C
          c₁ : CategoryTheory.Limits.Fan f₁
          c₂ : CategoryTheory.Limits.Fan f₂
          bc : CategoryTheory.Limits.BinaryFan c₁.pt c₂.pt
          h₁ : CategoryTheory.Limits.IsLimit c₁
          h₂ : CategoryTheory.Limits.IsLimit c₂
          h : CategoryTheory.Limits.IsLimit bc
          s : CategoryTheory.Limits.Fan (Sum.elim f₁ f₂)
          val✝ : ι₁
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun s => CategoryTheory.Limits.Fan. …
        -/
        /-
          case inl
          C : Type u₁
          inst✝ : CategoryTheory.Category.{u₂, u₁} C
          ι₁ : Type u_1
          ι₂ : Type u_2
          X : C
          f₁ : ι₁ → C
          f₂ : ι₂ → C
          c₁ : CategoryTheory.Limits.Fan f₁
          c₂ : CategoryTheory.Limits.Fan f₂
          bc : CategoryTheory.Limits.BinaryFan c₁.pt c₂.pt
          h₁ : CategoryTheory.Limits.IsLimit c₁
          h₂ : CategoryTheory.Limits.IsLimit c₂
          h : CategoryTheory.Limits.IsLimit bc
          s : CategoryTheory.Limits.Fan (Sum.elim f₁ f₂)
          val✝ : ι₁
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Fan.IsLimit.de …
        -/
        /-
          case inl
          C : Type u₁
          inst✝ : CategoryTheory.Category.{u₂, u₁} C
          ι₁ : Type u_1
          ι₂ : Type u_2
          X : C
          f₁ : ι₁ → C
          f₂ : ι₂ → C
          c₁ : CategoryTheory.Limits.Fan f₁
          c₂ : CategoryTheory.Limits.Fan f₂
          bc : CategoryTheory.Limits.BinaryFan c₁.pt c₂.pt
          h₁ : CategoryTheory.Limits.IsLimit c₁
          h₂ : CategoryTheory.Limits.IsLimit c₂
          h : CategoryTheory.Limits.IsLimit bc
          s : CategoryTheory.Limits.Fan (Sum.elim f₁ f₂)
          val✝ : ι₁
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Limits.Fan.mk s.pt f …
        -/
        /-
          🎉 no goals
        -/
        erw [← Category.assoc, h.fac]
        /-
          case inr
          C : Type u₁
          inst✝ : CategoryTheory.Category.{u₂, u₁} C
          ι₁ : Type u_1
          ι₂ : Type u_2
          X : C
          f₁ : ι₁ → C
          f₂ : ι₂ → C
          c₁ : CategoryTheory.Limits.Fan f₁
          c₂ : CategoryTheory.Limits.Fan f₂
          bc : CategoryTheory.Limits.BinaryFan c₁.pt c₂.pt
          h₁ : CategoryTheory.Limits.IsLimit c₁
          h₂ : CategoryTheory.Limits.IsLimit c₂
          h : CategoryTheory.Limits.IsLimit bc
          s : CategoryTheory.Limits.Fan (Sum.elim f₁ f₂)
          val✝ : ι₂
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Limits.Fan.mk s.pt f …
        -/
        simp only [pair_obj_left, mk_pt, mk_π_app, IsLimit.fac])
        /-
          🎉 no goals
        -/
    (fun s m hm ↦ Fan.IsLimit.hom_ext h _ _ <| fun w ↦ by
      /-
        C : Type u₁
        inst✝ : CategoryTheory.Category.{u₂, u₁} C
        ι₁ : Type u_1
        ι₂ : Type u_2
        X : C
        f₁ : ι₁ → C
        f₂ : ι₂ → C
        c₁ : CategoryTheory.Limits.Fan f₁
        c₂ : CategoryTheory.Limits.Fan f₂
        bc : CategoryTheory.Limits.BinaryFan c₁.pt c₂.pt
        h₁ : CategoryTheory.Limits.IsLimit c₁
        h₂ : CategoryTheory.Limits.IsLimit c₂
        h : CategoryTheory.Limits.IsLimit bc
        s : CategoryTheory.Limits.Fan (Sum.elim f₁ f₂)
        m : Quiver.Hom s.pt (CategoryTheory.Limits.Fan.mk bc.pt (c₁.combPairHoms c₂ bc …
        hm : ∀ (j : Sum ι₁ ι₂), Eq (CategoryTheory.CategoryStruct.comp m ((CategoryThe …
        w : CategoryTheory.Limits.WalkingPair
        ⊢ Eq (CategoryTheory.CategoryStruct.comp m (CategoryTheory.Limits.Fan.proj bc  …
      -/
      cases w
        /-
          case left
          C : Type u₁
          inst✝ : CategoryTheory.Category.{u₂, u₁} C
          ι₁ : Type u_1
          ι₂ : Type u_2
          X : C
          f₁ : ι₁ → C
          f₂ : ι₂ → C
          c₁ : CategoryTheory.Limits.Fan f₁
          c₂ : CategoryTheory.Limits.Fan f₂
          bc : CategoryTheory.Limits.BinaryFan c₁.pt c₂.pt
          h₁ : CategoryTheory.Limits.IsLimit c₁
          h₂ : CategoryTheory.Limits.IsLimit c₂
          h : CategoryTheory.Limits.IsLimit bc
          s : CategoryTheory.Limits.Fan (Sum.elim f₁ f₂)
          m : Quiver.Hom s.pt (CategoryTheory.Limits.Fan.mk bc.pt (c₁.combPairHoms c₂ bc …
          hm : ∀ (j : Sum ι₁ ι₂), Eq (CategoryTheory.CategoryStruct.comp m ((CategoryThe …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp m (CategoryTheory.Limits.Fan.proj bc  …
        -/
      · refine Fan.IsLimit.hom_ext h₁ _ _ (fun a ↦ by aesop)
        /-
          🎉 no goals
        -/
        /-
          case right
          C : Type u₁
          inst✝ : CategoryTheory.Category.{u₂, u₁} C
          ι₁ : Type u_1
          ι₂ : Type u_2
          X : C
          f₁ : ι₁ → C
          f₂ : ι₂ → C
          c₁ : CategoryTheory.Limits.Fan f₁
          c₂ : CategoryTheory.Limits.Fan f₂
          bc : CategoryTheory.Limits.BinaryFan c₁.pt c₂.pt
          h₁ : CategoryTheory.Limits.IsLimit c₁
          h₂ : CategoryTheory.Limits.IsLimit c₂
          h : CategoryTheory.Limits.IsLimit bc
          s : CategoryTheory.Limits.Fan (Sum.elim f₁ f₂)
          m : Quiver.Hom s.pt (CategoryTheory.Limits.Fan.mk bc.pt (c₁.combPairHoms c₂ bc …
          hm : ∀ (j : Sum ι₁ ι₂), Eq (CategoryTheory.CategoryStruct.comp m ((CategoryThe …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp m (CategoryTheory.Limits.Fan.proj bc  …
        -/
      · refine Fan.IsLimit.hom_ext h₂ _ _ (fun a ↦ by aesop))
        /-
          🎉 no goals
        -/


/-- For cofans on maps `f₁ : ι₁ → C`, `f₂ : ι₂ → C` and a binary cofan on their
cocone points, construct one family of morphisms indexed by `ι₁ ⊕ ι₂` -/
@[simp]
abbrev combPairHoms : (i : ι₁ ⊕ ι₂) → Sum.elim f₁ f₂ i ⟶ bc.pt
  | .inl a => c₁.inj a ≫ bc.inl
  | .inr a => c₂.inj a ≫ bc.inr


/-- If `c₁` and `c₂` are colimit cofans and `bc` is a colimit binary cofan on their cocone
points, then the cofan constructed from `combPairHoms` is a colimit cocone. -/
def combPairIsColimit : IsColimit (Cofan.mk bc.pt (combPairHoms c₁ c₂ bc)) :=
  mkCofanColimit _
    (fun s ↦ Cofan.IsColimit.desc h <| fun i ↦ by
      /-
        C : Type u₁
        inst✝ : CategoryTheory.Category.{u₂, u₁} C
        ι₁ : Type u_1
        ι₂ : Type u_2
        X : C
        f₁ : ι₁ → C
        f₂ : ι₂ → C
        c₁ : CategoryTheory.Limits.Cofan f₁
        c₂ : CategoryTheory.Limits.Cofan f₂
        bc : CategoryTheory.Limits.BinaryCofan c₁.pt c₂.pt
        h₁ : CategoryTheory.Limits.IsColimit c₁
        h₂ : CategoryTheory.Limits.IsColimit c₂
        h : CategoryTheory.Limits.IsColimit bc
        s : CategoryTheory.Limits.Cofan (Sum.elim f₁ f₂)
        i : CategoryTheory.Limits.WalkingPair
        ⊢ Quiver.Hom (CategoryTheory.Limits.WalkingPair.casesOn i c₁.pt c₂.pt) s.pt
      -/
      cases i
        /-
          case left
          C : Type u₁
          inst✝ : CategoryTheory.Category.{u₂, u₁} C
          ι₁ : Type u_1
          ι₂ : Type u_2
          X : C
          f₁ : ι₁ → C
          f₂ : ι₂ → C
          c₁ : CategoryTheory.Limits.Cofan f₁
          c₂ : CategoryTheory.Limits.Cofan f₂
          bc : CategoryTheory.Limits.BinaryCofan c₁.pt c₂.pt
          h₁ : CategoryTheory.Limits.IsColimit c₁
          h₂ : CategoryTheory.Limits.IsColimit c₂
          h : CategoryTheory.Limits.IsColimit bc
          s : CategoryTheory.Limits.Cofan (Sum.elim f₁ f₂)
          ⊢ Quiver.Hom (CategoryTheory.Limits.WalkingPair.casesOn CategoryTheory.Limits. …
        -/
      · exact Cofan.IsColimit.desc h₁ (fun a ↦ s.inj (.inl a))
        /-
          🎉 no goals
        -/
        /-
          case right
          C : Type u₁
          inst✝ : CategoryTheory.Category.{u₂, u₁} C
          ι₁ : Type u_1
          ι₂ : Type u_2
          X : C
          f₁ : ι₁ → C
          f₂ : ι₂ → C
          c₁ : CategoryTheory.Limits.Cofan f₁
          c₂ : CategoryTheory.Limits.Cofan f₂
          bc : CategoryTheory.Limits.BinaryCofan c₁.pt c₂.pt
          h₁ : CategoryTheory.Limits.IsColimit c₁
          h₂ : CategoryTheory.Limits.IsColimit c₂
          h : CategoryTheory.Limits.IsColimit bc
          s : CategoryTheory.Limits.Cofan (Sum.elim f₁ f₂)
          ⊢ Quiver.Hom (CategoryTheory.Limits.WalkingPair.casesOn CategoryTheory.Limits. …
        -/
      · exact Cofan.IsColimit.desc h₂ (fun a ↦ s.inj (.inr a)))
        /-
          🎉 no goals
        -/
    (fun s w ↦ by
      /-
        C : Type u₁
        inst✝ : CategoryTheory.Category.{u₂, u₁} C
        ι₁ : Type u_1
        ι₂ : Type u_2
        X : C
        f₁ : ι₁ → C
        f₂ : ι₂ → C
        c₁ : CategoryTheory.Limits.Cofan f₁
        c₂ : CategoryTheory.Limits.Cofan f₂
        bc : CategoryTheory.Limits.BinaryCofan c₁.pt c₂.pt
        h₁ : CategoryTheory.Limits.IsColimit c₁
        h₂ : CategoryTheory.Limits.IsColimit c₂
        h : CategoryTheory.Limits.IsColimit bc
        s : CategoryTheory.Limits.Cofan (Sum.elim f₁ f₂)
        w : Sum ι₁ ι₂
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Limits.Cofan.mk bc.p …
      -/
      cases w <;>
        /-
          case inl
          C : Type u₁
          inst✝ : CategoryTheory.Category.{u₂, u₁} C
          ι₁ : Type u_1
          ι₂ : Type u_2
          X : C
          f₁ : ι₁ → C
          f₂ : ι₂ → C
          c₁ : CategoryTheory.Limits.Cofan f₁
          c₂ : CategoryTheory.Limits.Cofan f₂
          bc : CategoryTheory.Limits.BinaryCofan c₁.pt c₂.pt
          h₁ : CategoryTheory.Limits.IsColimit c₁
          h₂ : CategoryTheory.Limits.IsColimit c₂
          h : CategoryTheory.Limits.IsColimit bc
          s : CategoryTheory.Limits.Cofan (Sum.elim f₁ f₂)
          val✝ : ι₁
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Limits.Cofan.mk bc.p …
        -/
        /-
          case inl
          C : Type u₁
          inst✝ : CategoryTheory.Category.{u₂, u₁} C
          ι₁ : Type u_1
          ι₂ : Type u_2
          X : C
          f₁ : ι₁ → C
          f₂ : ι₂ → C
          c₁ : CategoryTheory.Limits.Cofan f₁
          c₂ : CategoryTheory.Limits.Cofan f₂
          bc : CategoryTheory.Limits.BinaryCofan c₁.pt c₂.pt
          h₁ : CategoryTheory.Limits.IsColimit c₁
          h₂ : CategoryTheory.Limits.IsColimit c₂
          h : CategoryTheory.Limits.IsColimit bc
          s : CategoryTheory.Limits.Cofan (Sum.elim f₁ f₂)
          val✝ : ι₁
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (c₁.inj val✝) (CategoryTheory.Categor …
        -/
        /-
          case inl
          C : Type u₁
          inst✝ : CategoryTheory.Category.{u₂, u₁} C
          ι₁ : Type u_1
          ι₂ : Type u_2
          X : C
          f₁ : ι₁ → C
          f₂ : ι₂ → C
          c₁ : CategoryTheory.Limits.Cofan f₁
          c₂ : CategoryTheory.Limits.Cofan f₂
          bc : CategoryTheory.Limits.BinaryCofan c₁.pt c₂.pt
          h₁ : CategoryTheory.Limits.IsColimit c₁
          h₂ : CategoryTheory.Limits.IsColimit c₂
          h : CategoryTheory.Limits.IsColimit bc
          s : CategoryTheory.Limits.Cofan (Sum.elim f₁ f₂)
          val✝ : ι₁
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (c₁.inj val✝) ((CategoryTheory.Limits …
        -/
        /-
          🎉 no goals
        -/
        erw [h.fac]
        /-
          case inr
          C : Type u₁
          inst✝ : CategoryTheory.Category.{u₂, u₁} C
          ι₁ : Type u_1
          ι₂ : Type u_2
          X : C
          f₁ : ι₁ → C
          f₂ : ι₂ → C
          c₁ : CategoryTheory.Limits.Cofan f₁
          c₂ : CategoryTheory.Limits.Cofan f₂
          bc : CategoryTheory.Limits.BinaryCofan c₁.pt c₂.pt
          h₁ : CategoryTheory.Limits.IsColimit c₁
          h₂ : CategoryTheory.Limits.IsColimit c₂
          h : CategoryTheory.Limits.IsColimit bc
          s : CategoryTheory.Limits.Cofan (Sum.elim f₁ f₂)
          val✝ : ι₂
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (c₂.inj val✝) ((CategoryTheory.Limits …
        -/
        simp only [Cofan.mk_ι_app, Cofan.IsColimit.fac])
        /-
          🎉 no goals
        -/
    (fun s m hm ↦ Cofan.IsColimit.hom_ext h _ _ <| fun w ↦ by
      /-
        C : Type u₁
        inst✝ : CategoryTheory.Category.{u₂, u₁} C
        ι₁ : Type u_1
        ι₂ : Type u_2
        X : C
        f₁ : ι₁ → C
        f₂ : ι₂ → C
        c₁ : CategoryTheory.Limits.Cofan f₁
        c₂ : CategoryTheory.Limits.Cofan f₂
        bc : CategoryTheory.Limits.BinaryCofan c₁.pt c₂.pt
        h₁ : CategoryTheory.Limits.IsColimit c₁
        h₂ : CategoryTheory.Limits.IsColimit c₂
        h : CategoryTheory.Limits.IsColimit bc
        s : CategoryTheory.Limits.Cofan (Sum.elim f₁ f₂)
        m : Quiver.Hom (CategoryTheory.Limits.Cofan.mk bc.pt (c₁.combPairHoms c₂ bc)). …
        hm : ∀ (j : Sum ι₁ ι₂), Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheor …
        w : CategoryTheory.Limits.WalkingPair
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofan.inj bc w …
      -/
      cases w
        /-
          case left
          C : Type u₁
          inst✝ : CategoryTheory.Category.{u₂, u₁} C
          ι₁ : Type u_1
          ι₂ : Type u_2
          X : C
          f₁ : ι₁ → C
          f₂ : ι₂ → C
          c₁ : CategoryTheory.Limits.Cofan f₁
          c₂ : CategoryTheory.Limits.Cofan f₂
          bc : CategoryTheory.Limits.BinaryCofan c₁.pt c₂.pt
          h₁ : CategoryTheory.Limits.IsColimit c₁
          h₂ : CategoryTheory.Limits.IsColimit c₂
          h : CategoryTheory.Limits.IsColimit bc
          s : CategoryTheory.Limits.Cofan (Sum.elim f₁ f₂)
          m : Quiver.Hom (CategoryTheory.Limits.Cofan.mk bc.pt (c₁.combPairHoms c₂ bc)). …
          hm : ∀ (j : Sum ι₁ ι₂), Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheor …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofan.inj bc C …
        -/
      · refine Cofan.IsColimit.hom_ext h₁ _ _ (fun a ↦ by aesop)
        /-
          🎉 no goals
        -/
        /-
          case right
          C : Type u₁
          inst✝ : CategoryTheory.Category.{u₂, u₁} C
          ι₁ : Type u_1
          ι₂ : Type u_2
          X : C
          f₁ : ι₁ → C
          f₂ : ι₂ → C
          c₁ : CategoryTheory.Limits.Cofan f₁
          c₂ : CategoryTheory.Limits.Cofan f₂
          bc : CategoryTheory.Limits.BinaryCofan c₁.pt c₂.pt
          h₁ : CategoryTheory.Limits.IsColimit c₁
          h₂ : CategoryTheory.Limits.IsColimit c₂
          h : CategoryTheory.Limits.IsColimit bc
          s : CategoryTheory.Limits.Cofan (Sum.elim f₁ f₂)
          m : Quiver.Hom (CategoryTheory.Limits.Cofan.mk bc.pt (c₁.combPairHoms c₂ bc)). …
          hm : ∀ (j : Sum ι₁ ι₂), Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheor …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofan.inj bc C …
        -/
      · refine Cofan.IsColimit.hom_ext h₂ _ _ (fun a ↦ by aesop))
        /-
          🎉 no goals
        -/


