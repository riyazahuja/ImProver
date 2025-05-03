/-- This condition expresses that inclusion morphisms into coproducts are monomorphisms. -/
class MonoCoprod : Prop where
  /-- the left inclusion of a colimit binary cofan is mono -/
  binaryCofan_inl : ∀ ⦃A B : C⦄ (c : BinaryCofan A B) (_ : IsColimit c), Mono c.inl


instance (priority := 100) monoCoprodOfHasZeroMorphisms [HasZeroMorphisms C] : MonoCoprod C :=
  ⟨fun A B c hc => by
    haveI : IsSplitMono c.inl :=
      IsSplitMono.mk' (SplitMono.mk (hc.desc (BinaryCofan.mk (𝟙 A) 0)) (IsColimit.fac _ _ _))
    /-
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      A B : C
      c : CategoryTheory.Limits.BinaryCofan A B
      hc : CategoryTheory.Limits.IsColimit c
      this : CategoryTheory.IsSplitMono c.inl
      ⊢ CategoryTheory.Mono c.inl
    -/
    infer_instance⟩
    /-
      🎉 no goals
    -/


theorem binaryCofan_inr {A B : C} [MonoCoprod C] (c : BinaryCofan A B) (hc : IsColimit c) :
    Mono c.inr := by
  haveI hc' : IsColimit (BinaryCofan.mk c.inr c.inl) :=
    BinaryCofan.IsColimit.mk _ (fun f₁ f₂ => hc.desc (BinaryCofan.mk f₂ f₁))
      (by aesop_cat) (by aesop_cat)
      (fun f₁ f₂ m h₁ h₂ => BinaryCofan.IsColimit.hom_ext hc (by aesop_cat) (by aesop_cat))
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    A B : C
    inst✝ : CategoryTheory.Limits.MonoCoprod C
    c : CategoryTheory.Limits.BinaryCofan A B
    hc : CategoryTheory.Limits.IsColimit c
    hc' : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.BinaryCofan.mk c. …
    ⊢ CategoryTheory.Mono c.inr
  -/
  exact binaryCofan_inl _ hc'
  /-
    🎉 no goals
  -/


instance {A B : C} [MonoCoprod C] [HasBinaryCoproduct A B] : Mono (coprod.inl : A ⟶ A ⨿ B) :=
  binaryCofan_inl _ (colimit.isColimit _)


instance {A B : C} [MonoCoprod C] [HasBinaryCoproduct A B] : Mono (coprod.inr : B ⟶ A ⨿ B) :=
  binaryCofan_inr _ (colimit.isColimit _)


theorem mono_inl_iff {A B : C} {c₁ c₂ : BinaryCofan A B} (hc₁ : IsColimit c₁) (hc₂ : IsColimit c₂) :
    Mono c₁.inl ↔ Mono c₂.inl := by
  suffices
    ∀ (c₁ c₂ : BinaryCofan A B) (_ : IsColimit c₁) (_ : IsColimit c₂) (_ : Mono c₁.inl),
      Mono c₂.inl
    by exact ⟨fun h₁ => this _ _ hc₁ hc₂ h₁, fun h₂ => this _ _ hc₂ hc₁ h₂⟩
  /-
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    A B : C
    c₁ c₂ : CategoryTheory.Limits.BinaryCofan A B
    hc₁ : CategoryTheory.Limits.IsColimit c₁
    hc₂ : CategoryTheory.Limits.IsColimit c₂
    ⊢ ∀ (c₁ c₂ : CategoryTheory.Limits.BinaryCofan A B), CategoryTheory.Limits.IsC …
  -/
  intro c₁ c₂ hc₁ hc₂
  /-
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    A B : C
    c₁✝ c₂✝ : CategoryTheory.Limits.BinaryCofan A B
    hc₁✝ : CategoryTheory.Limits.IsColimit c₁✝
    hc₂✝ : CategoryTheory.Limits.IsColimit c₂✝
    c₁ c₂ : CategoryTheory.Limits.BinaryCofan A B
    hc₁ : CategoryTheory.Limits.IsColimit c₁
    hc₂ : CategoryTheory.Limits.IsColimit c₂
    ⊢ CategoryTheory.Mono c₁.inl → CategoryTheory.Mono c₂.inl
  -/
  intro
  simpa only [IsColimit.comp_coconePointUniqueUpToIso_hom] using
    mono_comp c₁.inl (hc₁.coconePointUniqueUpToIso hc₂).hom


theorem mk' (h : ∀ A B : C, ∃ (c : BinaryCofan A B) (_ : IsColimit c), Mono c.inl) : MonoCoprod C :=
  ⟨fun A B c' hc' => by
    /-
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} C
      h : ∀ (A B : C), Exists fun c => Exists fun x => CategoryTheory.Mono c.inl
      A B : C
      c' : CategoryTheory.Limits.BinaryCofan A B
      hc' : CategoryTheory.Limits.IsColimit c'
      ⊢ CategoryTheory.Mono c'.inl
    -/
    obtain ⟨c, hc₁, hc₂⟩ := h A B
    /-
      case intro.intro
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} C
      h : ∀ (A B : C), Exists fun c => Exists fun x => CategoryTheory.Mono c.inl
      A B : C
      c' : CategoryTheory.Limits.BinaryCofan A B
      hc' : CategoryTheory.Limits.IsColimit c'
      c : CategoryTheory.Limits.BinaryCofan A B
      hc₁ : CategoryTheory.Limits.IsColimit c
      hc₂ : CategoryTheory.Mono c.inl
      ⊢ CategoryTheory.Mono c'.inl
    -/
    simpa only [mono_inl_iff hc' hc₁] using hc₂⟩
    /-
      🎉 no goals
    -/


instance monoCoprodType : MonoCoprod (Type u) :=
  MonoCoprod.mk' fun A B => by
    /-
      C : Type u_1
      inst✝ : CategoryTheory.Category.{?u.12629, u_1} C
      A B : Type u
      ⊢ Exists fun c => Exists fun x => CategoryTheory.Mono c.inl
    -/
    refine ⟨BinaryCofan.mk (Sum.inl : A ⟶ A ⊕ B) Sum.inr, ?_, ?_⟩
    · exact BinaryCofan.IsColimit.mk _
        (fun f₁ f₂ x => by
          rcases x with x | x
          exacts [f₁ x, f₂ x])
        (fun f₁ f₂ => by rfl)
        (fun f₁ f₂ => by rfl)
        (fun f₁ f₂ m h₁ h₂ => by
          funext x
          rcases x with x | x
          · exact congr_fun h₁ x
          · exact congr_fun h₂ x)
      /-
        case refine_2
        C : Type u_1
        inst✝ : CategoryTheory.Category.{?u.12629, u_1} C
        A B : Type u
        ⊢ CategoryTheory.Mono (CategoryTheory.Limits.BinaryCofan.mk Sum.inl Sum.inr).inl
      -/
    · rw [mono_iff_injective]
      /-
        case refine_2
        C : Type u_1
        inst✝ : CategoryTheory.Category.{?u.12629, u_1} C
        A B : Type u
        ⊢ Function.Injective (CategoryTheory.Limits.BinaryCofan.mk Sum.inl Sum.inr).inl
      -/
      intro a₁ a₂ h
      /-
        case refine_2
        C : Type u_1
        inst✝ : CategoryTheory.Category.{?u.12629, u_1} C
        A B : Type u
        a₁ a₂ : (CategoryTheory.Limits.pair A B).obj { as := CategoryTheory.Limits.Wal …
        h : Eq ((CategoryTheory.Limits.BinaryCofan.mk Sum.inl Sum.inr).inl a₁) ((Categ …
        ⊢ Eq a₁ a₂
      -/
      simpa using h
      /-
        🎉 no goals
      -/


/-- Given a family of objects `X : I₁ ⊕ I₂ → C`, a cofan of `X`, and two colimit cofans
of `X ∘ Sum.inl` and `X ∘ Sum.inr`, this is a cofan for `c₁.pt` and `c₂.pt` whose
point is `c.pt`. -/
@[simp]
def binaryCofanSum : BinaryCofan c₁.pt c₂.pt :=
  BinaryCofan.mk (Cofan.IsColimit.desc hc₁ (fun i₁ => c.inj (Sum.inl i₁)))
    (Cofan.IsColimit.desc hc₂ (fun i₂ => c.inj (Sum.inr i₂)))


/-- The binary cofan `binaryCofanSum c c₁ c₂ hc₁ hc₂` is colimit. -/
def isColimitBinaryCofanSum : IsColimit (binaryCofanSum c c₁ c₂ hc₁ hc₂) :=
  BinaryCofan.IsColimit.mk _ (fun f₁ f₂ => Cofan.IsColimit.desc hc (fun i => match i with
      | Sum.inl i₁ => c₁.inj i₁ ≫ f₁
      | Sum.inr i₂ => c₂.inj i₂ ≫ f₂))
                                                      /-
                                                        C : Type u_1
                                                        inst✝ : CategoryTheory.Category.{?u.16578, u_1} C
                                                        I₁ : Type u_2
                                                        I₂ : Type u_3
                                                        X : Sum I₁ I₂ → C
                                                        c : CategoryTheory.Limits.Cofan X
                                                        c₁ : CategoryTheory.Limits.Cofan (Function.comp X Sum.inl)
                                                        c₂ : CategoryTheory.Limits.Cofan (Function.comp X Sum.inr)
                                                        hc : CategoryTheory.Limits.IsColimit c
                                                        hc₁ : CategoryTheory.Limits.IsColimit c₁
                                                        hc₂ : CategoryTheory.Limits.IsColimit c₂
                                                        T✝ : C
                                                        f₁ : Quiver.Hom c₁.pt T✝
                                                        f₂ : Quiver.Hom c₂.pt T✝
                                                        ⊢ ∀ (i : I₁), Eq (CategoryTheory.CategoryStruct.comp (c₁.inj i) (CategoryTheor …
                                                      -/
    (fun f₁ f₂ => Cofan.IsColimit.hom_ext hc₁ _ _ (by simp))
                                                      /-
                                                        🎉 no goals
                                                      -/
                                                      /-
                                                        C : Type u_1
                                                        inst✝ : CategoryTheory.Category.{?u.16578, u_1} C
                                                        I₁ : Type u_2
                                                        I₂ : Type u_3
                                                        X : Sum I₁ I₂ → C
                                                        c : CategoryTheory.Limits.Cofan X
                                                        c₁ : CategoryTheory.Limits.Cofan (Function.comp X Sum.inl)
                                                        c₂ : CategoryTheory.Limits.Cofan (Function.comp X Sum.inr)
                                                        hc : CategoryTheory.Limits.IsColimit c
                                                        hc₁ : CategoryTheory.Limits.IsColimit c₁
                                                        hc₂ : CategoryTheory.Limits.IsColimit c₂
                                                        T✝ : C
                                                        f₁ : Quiver.Hom c₁.pt T✝
                                                        f₂ : Quiver.Hom c₂.pt T✝
                                                        ⊢ ∀ (i : I₂), Eq (CategoryTheory.CategoryStruct.comp (c₂.inj i) (CategoryTheor …
                                                      -/
    (fun f₁ f₂ => Cofan.IsColimit.hom_ext hc₂ _ _ (by simp))
                                                      /-
                                                        🎉 no goals
                                                      -/
    (fun f₁ f₂ m hm₁ hm₂ => by
      /-
        C : Type u_1
        inst✝ : CategoryTheory.Category.{?u.16578, u_1} C
        I₁ : Type u_2
        I₂ : Type u_3
        X : Sum I₁ I₂ → C
        c : CategoryTheory.Limits.Cofan X
        c₁ : CategoryTheory.Limits.Cofan (Function.comp X Sum.inl)
        c₂ : CategoryTheory.Limits.Cofan (Function.comp X Sum.inr)
        hc : CategoryTheory.Limits.IsColimit c
        hc₁ : CategoryTheory.Limits.IsColimit c₁
        hc₂ : CategoryTheory.Limits.IsColimit c₂
        T✝ : C
        f₁ : Quiver.Hom c₁.pt T✝
        f₂ : Quiver.Hom c₂.pt T✝
        m : Quiver.Hom (CategoryTheory.Limits.MonoCoprod.binaryCofanSum c c₁ c₂ hc₁ hc …
        hm₁ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.MonoCoprod …
        hm₂ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.MonoCoprod …
        ⊢ Eq m ((fun {T} f₁ f₂ => CategoryTheory.Limits.Cofan.IsColimit.desc hc fun i  …
      -/
      apply Cofan.IsColimit.hom_ext hc
      /-
        case h
        C : Type u_1
        inst✝ : CategoryTheory.Category.{?u.16578, u_1} C
        I₁ : Type u_2
        I₂ : Type u_3
        X : Sum I₁ I₂ → C
        c : CategoryTheory.Limits.Cofan X
        c₁ : CategoryTheory.Limits.Cofan (Function.comp X Sum.inl)
        c₂ : CategoryTheory.Limits.Cofan (Function.comp X Sum.inr)
        hc : CategoryTheory.Limits.IsColimit c
        hc₁ : CategoryTheory.Limits.IsColimit c₁
        hc₂ : CategoryTheory.Limits.IsColimit c₂
        T✝ : C
        f₁ : Quiver.Hom c₁.pt T✝
        f₂ : Quiver.Hom c₂.pt T✝
        m : Quiver.Hom (CategoryTheory.Limits.MonoCoprod.binaryCofanSum c c₁ c₂ hc₁ hc …
        hm₁ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.MonoCoprod …
        hm₂ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.MonoCoprod …
        ⊢ ∀ (i : Sum I₁ I₂), Eq (CategoryTheory.CategoryStruct.comp (c.inj i) m) (Cate …
      -/
                         /-
                           🎉 no goals
                         -/
      rintro (i₁|i₂) <;> aesop_cat)
                         /-
                           🎉 no goals
                         -/


lemma mono_binaryCofanSum_inl [MonoCoprod C] :
    Mono (binaryCofanSum c c₁ c₂ hc₁ hc₂).inl :=
  MonoCoprod.binaryCofan_inl _ (isColimitBinaryCofanSum c c₁ c₂ hc hc₁ hc₂)


lemma mono_binaryCofanSum_inr [MonoCoprod C] :
    Mono (binaryCofanSum c c₁ c₂ hc₁ hc₂).inr :=
  MonoCoprod.binaryCofan_inr _ (isColimitBinaryCofanSum c c₁ c₂ hc hc₁ hc₂)


lemma mono_binaryCofanSum_inl' [MonoCoprod C] (inl : c₁.pt ⟶ c.pt)
    (hinl : ∀ (i₁ : I₁), c₁.inj i₁ ≫ inl = c.inj (Sum.inl i₁)) :
    Mono inl := by
  suffices inl = (binaryCofanSum c c₁ c₂ hc₁ hc₂).inl by
    rw [this]
    exact MonoCoprod.binaryCofan_inl _ (isColimitBinaryCofanSum c c₁ c₂ hc hc₁ hc₂)
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_4, u_1} C
    I₁ : Type u_2
    I₂ : Type u_3
    X : Sum I₁ I₂ → C
    c : CategoryTheory.Limits.Cofan X
    c₁ : CategoryTheory.Limits.Cofan (Function.comp X Sum.inl)
    c₂ : CategoryTheory.Limits.Cofan (Function.comp X Sum.inr)
    hc : CategoryTheory.Limits.IsColimit c
    hc₁ : CategoryTheory.Limits.IsColimit c₁
    hc₂ : CategoryTheory.Limits.IsColimit c₂
    inst✝ : CategoryTheory.Limits.MonoCoprod C
    inl : Quiver.Hom c₁.pt c.pt
    hinl : ∀ (i₁ : I₁), Eq (CategoryTheory.CategoryStruct.comp (c₁.inj i₁) inl) (c …
    ⊢ Eq inl (CategoryTheory.Limits.MonoCoprod.binaryCofanSum c c₁ c₂ hc₁ hc₂).inl
  -/
  exact Cofan.IsColimit.hom_ext hc₁ _ _ (by simpa using hinl)
  /-
    🎉 no goals
  -/


lemma mono_binaryCofanSum_inr' [MonoCoprod C] (inr : c₂.pt ⟶ c.pt)
    (hinr : ∀ (i₂ : I₂), c₂.inj i₂ ≫ inr = c.inj (Sum.inr i₂)) :
    Mono inr := by
  suffices inr = (binaryCofanSum c c₁ c₂ hc₁ hc₂).inr by
    rw [this]
    exact MonoCoprod.binaryCofan_inr _ (isColimitBinaryCofanSum c c₁ c₂ hc hc₁ hc₂)
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_4, u_1} C
    I₁ : Type u_2
    I₂ : Type u_3
    X : Sum I₁ I₂ → C
    c : CategoryTheory.Limits.Cofan X
    c₁ : CategoryTheory.Limits.Cofan (Function.comp X Sum.inl)
    c₂ : CategoryTheory.Limits.Cofan (Function.comp X Sum.inr)
    hc : CategoryTheory.Limits.IsColimit c
    hc₁ : CategoryTheory.Limits.IsColimit c₁
    hc₂ : CategoryTheory.Limits.IsColimit c₂
    inst✝ : CategoryTheory.Limits.MonoCoprod C
    inr : Quiver.Hom c₂.pt c.pt
    hinr : ∀ (i₂ : I₂), Eq (CategoryTheory.CategoryStruct.comp (c₂.inj i₂) inr) (c …
    ⊢ Eq inr (CategoryTheory.Limits.MonoCoprod.binaryCofanSum c c₁ c₂ hc₁ hc₂).inr
  -/
  exact Cofan.IsColimit.hom_ext hc₂ _ _ (by simpa using hinr)
  /-
    🎉 no goals
  -/


lemma mono_of_injective_aux (hι : Function.Injective ι) (c : Cofan X) (c₁ : Cofan (X ∘ ι))
    (hc : IsColimit c) (hc₁ : IsColimit c₁)
    (c₂ : Cofan (fun (k : ((Set.range ι)ᶜ : Set I)) => X k.1))
    (hc₂ : IsColimit c₂) : Mono (Cofan.IsColimit.desc hc₁ (fun i => c.inj (ι i))) := by
  classical
  let e := ((Equiv.ofInjective ι hι).sumCongr (Equiv.refl _)).trans (Equiv.Set.sumCompl _)
  refine mono_binaryCofanSum_inl' (Cofan.mk c.pt (fun i' => c.inj (e i'))) _ _ ?_
    hc₁ hc₂ _ (by simp [e])
  exact IsColimit.ofIsoColimit ((IsColimit.ofCoconeEquiv (Cocones.equivalenceOfReindexing
    (Discrete.equivalence e) (Iso.refl _))).symm hc) (Cocones.ext (Iso.refl _))


include hc in
lemma mono_of_injective [HasCoproduct (fun (k : ((Set.range ι)ᶜ : Set I)) => X k.1)] :
    Mono (Cofan.IsColimit.desc hc₁ (fun i => c.inj (ι i))) :=
  mono_of_injective_aux X ι hι c c₁ hc hc₁ _ (colimit.isColimit _)


lemma mono_of_injective' [HasCoproduct (X ∘ ι)] [HasCoproduct X]
    [HasCoproduct (fun (k : ((Set.range ι)ᶜ : Set I)) => X k.1)] :
    Mono (Sigma.desc (f := X ∘ ι) (fun j => Sigma.ι X (ι j))) :=
  mono_of_injective X ι hι _ _ (colimit.isColimit _) (colimit.isColimit _)


lemma mono_map'_of_injective [HasCoproduct (X ∘ ι)] [HasCoproduct X]
    [HasCoproduct (fun (k : ((Set.range ι)ᶜ : Set I)) => X k.1)] :
    Mono (Sigma.map' ι (fun j => 𝟙 ((X ∘ ι) j))) := by
  /-
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
    inst✝³ : CategoryTheory.Limits.MonoCoprod C
    I : Type u_2
    J : Type u_3
    X : I → C
    ι : J → I
    hι : Function.Injective ι
    inst✝² : CategoryTheory.Limits.HasCoproduct (Function.comp X ι)
    inst✝¹ : CategoryTheory.Limits.HasCoproduct X
    inst✝ : CategoryTheory.Limits.HasCoproduct fun k => X ↑k
    ⊢ CategoryTheory.Mono (CategoryTheory.Limits.Sigma.map' ι fun j => CategoryThe …
  -/
  convert mono_of_injective' X ι hι
  /-
    case h.e'_5
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
    inst✝³ : CategoryTheory.Limits.MonoCoprod C
    I : Type u_2
    J : Type u_3
    X : I → C
    ι : J → I
    hι : Function.Injective ι
    inst✝² : CategoryTheory.Limits.HasCoproduct (Function.comp X ι)
    inst✝¹ : CategoryTheory.Limits.HasCoproduct X
    inst✝ : CategoryTheory.Limits.HasCoproduct fun k => X ↑k
    ⊢ Eq (CategoryTheory.Limits.Sigma.map' ι fun j => CategoryTheory.CategoryStruc …
  -/
  apply Sigma.hom_ext
  /-
    case h.e'_5.h
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
    inst✝³ : CategoryTheory.Limits.MonoCoprod C
    I : Type u_2
    J : Type u_3
    X : I → C
    ι : J → I
    hι : Function.Injective ι
    inst✝² : CategoryTheory.Limits.HasCoproduct (Function.comp X ι)
    inst✝¹ : CategoryTheory.Limits.HasCoproduct X
    inst✝ : CategoryTheory.Limits.HasCoproduct fun k => X ↑k
    ⊢ ∀ (b : J), Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Sig …
  -/
  intro j
  /-
    case h.e'_5.h
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
    inst✝³ : CategoryTheory.Limits.MonoCoprod C
    I : Type u_2
    J : Type u_3
    X : I → C
    ι : J → I
    hι : Function.Injective ι
    inst✝² : CategoryTheory.Limits.HasCoproduct (Function.comp X ι)
    inst✝¹ : CategoryTheory.Limits.HasCoproduct X
    inst✝ : CategoryTheory.Limits.HasCoproduct fun k => X ↑k
    j : J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Sigma.ι (Funct …
  -/
  rw [Sigma.ι_comp_map', id_comp, colimit.ι_desc]
  /-
    case h.e'_5.h
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
    inst✝³ : CategoryTheory.Limits.MonoCoprod C
    I : Type u_2
    J : Type u_3
    X : I → C
    ι : J → I
    hι : Function.Injective ι
    inst✝² : CategoryTheory.Limits.HasCoproduct (Function.comp X ι)
    inst✝¹ : CategoryTheory.Limits.HasCoproduct X
    inst✝ : CategoryTheory.Limits.HasCoproduct fun k => X ↑k
    j : J
    ⊢ Eq (CategoryTheory.Limits.Sigma.ι X (ι j)) ((CategoryTheory.Limits.Cofan.mk  …
  -/
  simp
  /-
    🎉 no goals
  -/


lemma mono_inj (c : Cofan X) (h : IsColimit c) (i : I)
    [HasCoproduct (fun (k : ((Set.range (fun _ : Unit ↦ i))ᶜ : Set I)) => X k.1)] :
    Mono (Cofan.inj c i) := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_3, u_1} C
    inst✝¹ : CategoryTheory.Limits.MonoCoprod C
    I : Type u_2
    X : I → C
    c : CategoryTheory.Limits.Cofan X
    h : CategoryTheory.Limits.IsColimit c
    i : I
    inst✝ : CategoryTheory.Limits.HasCoproduct fun k => X ↑k
    ⊢ CategoryTheory.Mono (c.inj i)
  -/
  let ι : Unit → I := fun _ ↦ i
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_3, u_1} C
    inst✝¹ : CategoryTheory.Limits.MonoCoprod C
    I : Type u_2
    X : I → C
    c : CategoryTheory.Limits.Cofan X
    h : CategoryTheory.Limits.IsColimit c
    i : I
    inst✝ : CategoryTheory.Limits.HasCoproduct fun k => X ↑k
    ι : Unit → I := fun x => i
    ⊢ CategoryTheory.Mono (c.inj i)
  -/
  have hι : Function.Injective ι := fun _ _ _ ↦ rfl
  exact mono_of_injective X ι hι c (Cofan.mk (X i) (fun _ ↦ 𝟙 _)) h
    (mkCofanColimit _ (fun s => s.inj ()))


instance mono_ι [HasCoproduct X] (i : I)
    [HasCoproduct (fun (k : ((Set.range (fun _ : Unit ↦ i))ᶜ : Set I)) => X k.1)] :
    Mono (Sigma.ι X i) :=
  mono_inj X _ (colimit.isColimit _) i


theorem monoCoprod_of_preservesCoprod_of_reflectsMono [MonoCoprod D]
    [PreservesColimitsOfShape (Discrete WalkingPair) F]
    [ReflectsMonomorphisms F] : MonoCoprod C where
  binaryCofan_inl {A B} c h := by
    /-
      C : Type u_1
      inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
      D : Type u_2
      inst✝³ : CategoryTheory.Category.{u_3, u_2} D
      F : CategoryTheory.Functor C D
      inst✝² : CategoryTheory.Limits.MonoCoprod D
      inst✝¹ : CategoryTheory.Limits.PreservesColimitsOfShape (CategoryTheory.Discre …
      inst✝ : F.ReflectsMonomorphisms
      A B : C
      c : CategoryTheory.Limits.BinaryCofan A B
      h : CategoryTheory.Limits.IsColimit c
      ⊢ CategoryTheory.Mono c.inl
    -/
    let c' := BinaryCofan.mk (F.map c.inl) (F.map c.inr)
    /-
      C : Type u_1
      inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
      D : Type u_2
      inst✝³ : CategoryTheory.Category.{u_3, u_2} D
      F : CategoryTheory.Functor C D
      inst✝² : CategoryTheory.Limits.MonoCoprod D
      inst✝¹ : CategoryTheory.Limits.PreservesColimitsOfShape (CategoryTheory.Discre …
      inst✝ : F.ReflectsMonomorphisms
      A B : C
      c : CategoryTheory.Limits.BinaryCofan A B
      h : CategoryTheory.Limits.IsColimit c
      c' : CategoryTheory.Limits.BinaryCofan (F.obj ((CategoryTheory.Limits.pair A B …
      ⊢ CategoryTheory.Mono c.inl
    -/
    apply mono_of_mono_map F
    /-
      C : Type u_1
      inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
      D : Type u_2
      inst✝³ : CategoryTheory.Category.{u_3, u_2} D
      F : CategoryTheory.Functor C D
      inst✝² : CategoryTheory.Limits.MonoCoprod D
      inst✝¹ : CategoryTheory.Limits.PreservesColimitsOfShape (CategoryTheory.Discre …
      inst✝ : F.ReflectsMonomorphisms
      A B : C
      c : CategoryTheory.Limits.BinaryCofan A B
      h : CategoryTheory.Limits.IsColimit c
      c' : CategoryTheory.Limits.BinaryCofan (F.obj ((CategoryTheory.Limits.pair A B …
      ⊢ CategoryTheory.Mono (F.map c.inl)
    -/
    show Mono c'.inl
    /-
      C : Type u_1
      inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
      D : Type u_2
      inst✝³ : CategoryTheory.Category.{u_3, u_2} D
      F : CategoryTheory.Functor C D
      inst✝² : CategoryTheory.Limits.MonoCoprod D
      inst✝¹ : CategoryTheory.Limits.PreservesColimitsOfShape (CategoryTheory.Discre …
      inst✝ : F.ReflectsMonomorphisms
      A B : C
      c : CategoryTheory.Limits.BinaryCofan A B
      h : CategoryTheory.Limits.IsColimit c
      c' : CategoryTheory.Limits.BinaryCofan (F.obj ((CategoryTheory.Limits.pair A B …
      ⊢ CategoryTheory.Mono c'.inl
    -/
    apply MonoCoprod.binaryCofan_inl
    /-
      case x
      C : Type u_1
      inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
      D : Type u_2
      inst✝³ : CategoryTheory.Category.{u_3, u_2} D
      F : CategoryTheory.Functor C D
      inst✝² : CategoryTheory.Limits.MonoCoprod D
      inst✝¹ : CategoryTheory.Limits.PreservesColimitsOfShape (CategoryTheory.Discre …
      inst✝ : F.ReflectsMonomorphisms
      A B : C
      c : CategoryTheory.Limits.BinaryCofan A B
      h : CategoryTheory.Limits.IsColimit c
      c' : CategoryTheory.Limits.BinaryCofan (F.obj ((CategoryTheory.Limits.pair A B …
      ⊢ CategoryTheory.Limits.IsColimit c'
    -/
    apply mapIsColimitOfPreservesOfIsColimit F
    /-
      case x.l
      C : Type u_1
      inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
      D : Type u_2
      inst✝³ : CategoryTheory.Category.{u_3, u_2} D
      F : CategoryTheory.Functor C D
      inst✝² : CategoryTheory.Limits.MonoCoprod D
      inst✝¹ : CategoryTheory.Limits.PreservesColimitsOfShape (CategoryTheory.Discre …
      inst✝ : F.ReflectsMonomorphisms
      A B : C
      c : CategoryTheory.Limits.BinaryCofan A B
      h : CategoryTheory.Limits.IsColimit c
      c' : CategoryTheory.Limits.BinaryCofan (F.obj ((CategoryTheory.Limits.pair A B …
      ⊢ CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.BinaryCofan.mk c.inl  …
    -/
    apply IsColimit.ofIsoColimit h
    /-
      case x.l
      C : Type u_1
      inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
      D : Type u_2
      inst✝³ : CategoryTheory.Category.{u_3, u_2} D
      F : CategoryTheory.Functor C D
      inst✝² : CategoryTheory.Limits.MonoCoprod D
      inst✝¹ : CategoryTheory.Limits.PreservesColimitsOfShape (CategoryTheory.Discre …
      inst✝ : F.ReflectsMonomorphisms
      A B : C
      c : CategoryTheory.Limits.BinaryCofan A B
      h : CategoryTheory.Limits.IsColimit c
      c' : CategoryTheory.Limits.BinaryCofan (F.obj ((CategoryTheory.Limits.pair A B …
      ⊢ CategoryTheory.Iso c (CategoryTheory.Limits.BinaryCofan.mk c.inl c.inr)
    -/
    refine Cocones.ext (φ := eqToIso rfl) ?_
    /-
      case x.l
      C : Type u_1
      inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
      D : Type u_2
      inst✝³ : CategoryTheory.Category.{u_3, u_2} D
      F : CategoryTheory.Functor C D
      inst✝² : CategoryTheory.Limits.MonoCoprod D
      inst✝¹ : CategoryTheory.Limits.PreservesColimitsOfShape (CategoryTheory.Discre …
      inst✝ : F.ReflectsMonomorphisms
      A B : C
      c : CategoryTheory.Limits.BinaryCofan A B
      h : CategoryTheory.Limits.IsColimit c
      c' : CategoryTheory.Limits.BinaryCofan (F.obj ((CategoryTheory.Limits.pair A B …
      ⊢ ∀ (j : CategoryTheory.Discrete CategoryTheory.Limits.WalkingPair), Eq (Categ …
    -/
    rintro ⟨(j₁|j₂)⟩ <;> simp only [const_obj_obj, eqToIso_refl, Iso.refl_hom,
      Category.comp_id, BinaryCofan.mk_inl, BinaryCofan.mk_inr]


instance [ConcreteCategory C] [PreservesColimitsOfShape (Discrete WalkingPair) (forget C)]
    [ReflectsMonomorphisms (forget C)] : MonoCoprod C :=
  monoCoprod_of_preservesCoprod_of_reflectsMono (forget C)


