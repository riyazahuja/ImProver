@[nolint synTaut]
theorem evalComp_nil_nil {f g h : C} (α : f ≅ g) (β : g ≅ h) :
    (α ≪≫ β).hom = (α ≪≫ β).hom := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    f g h : C
    α : CategoryTheory.Iso f g
    β : CategoryTheory.Iso g h
    ⊢ Eq (α.trans β).hom (α.trans β).hom
  -/
  simp
  /-
    🎉 no goals
  -/


theorem evalComp_nil_cons {f g h i j : C} (α : f ≅ g) (β : g ≅ h) (η : h ⟶ i) (ηs : i ⟶ j) :
    α.hom ≫ (β.hom ≫ η ≫ ηs) = (α ≪≫ β).hom ≫ η ≫ ηs := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    f g h i j : C
    α : CategoryTheory.Iso f g
    β : CategoryTheory.Iso g h
    η : Quiver.Hom h i
    ηs : Quiver.Hom i j
    ⊢ Eq (CategoryTheory.CategoryStruct.comp α.hom (CategoryTheory.CategoryStruct. …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem evalComp_cons {f g h i j : C} (α : f ≅ g) (η : g ⟶ h) {ηs : h ⟶ i} {θ : i ⟶ j} {ι : h ⟶ j}
    (e_ι : ηs ≫ θ = ι)  :
    (α.hom ≫ η ≫ ηs) ≫ θ = α.hom ≫ η ≫ ι := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    f g h i j : C
    α : CategoryTheory.Iso f g
    η : Quiver.Hom g h
    ηs : Quiver.Hom h i
    θ : Quiver.Hom i j
    ι : Quiver.Hom h j
    e_ι : Eq (CategoryTheory.CategoryStruct.comp ηs θ) ι
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp α …
  -/
  simp [e_ι]
  /-
    🎉 no goals
  -/


theorem eval_comp
    {η η' : f ⟶ g} {θ θ' : g ⟶ h} {ι : f ⟶ h}
    (e_η : η = η') (e_θ : θ = θ') (e_ηθ : η' ≫ θ' = ι) :
    η ≫ θ = ι := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    f g h : C
    η η' : Quiver.Hom f g
    θ θ' : Quiver.Hom g h
    ι : Quiver.Hom f h
    e_η : Eq η η'
    e_θ : Eq θ θ'
    e_ηθ : Eq (CategoryTheory.CategoryStruct.comp η' θ') ι
    ⊢ Eq (CategoryTheory.CategoryStruct.comp η θ) ι
  -/
  simp [e_η, e_θ, e_ηθ]
  /-
    🎉 no goals
  -/


theorem eval_of (η : f ⟶ g) :
    η = (Iso.refl _).hom ≫ η ≫ (Iso.refl _).hom := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    f g : C
    η : Quiver.Hom f g
    ⊢ Eq η (CategoryTheory.CategoryStruct.comp (CategoryTheory.Iso.refl f).hom (Ca …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem eval_monoidalComp
    {η η' : f ⟶ g} {α : g ≅ h} {θ θ' : h ⟶ i} {αθ : g ⟶ i} {ηαθ : f ⟶ i}
    (e_η : η = η') (e_θ : θ = θ') (e_αθ : α.hom ≫ θ' = αθ) (e_ηαθ : η' ≫ αθ = ηαθ) :
    η ≫ α.hom ≫ θ = ηαθ := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    f g h i : C
    η η' : Quiver.Hom f g
    α : CategoryTheory.Iso g h
    θ θ' : Quiver.Hom h i
    αθ : Quiver.Hom g i
    ηαθ : Quiver.Hom f i
    e_η : Eq η η'
    e_θ : Eq θ θ'
    e_αθ : Eq (CategoryTheory.CategoryStruct.comp α.hom θ') αθ
    e_ηαθ : Eq (CategoryTheory.CategoryStruct.comp η' αθ) ηαθ
    ⊢ Eq (CategoryTheory.CategoryStruct.comp η (CategoryTheory.CategoryStruct.comp …
  -/
  simp [e_η, e_θ, e_αθ, e_ηαθ]
  /-
    🎉 no goals
  -/


@[nolint synTaut]
theorem evalWhiskerLeft_nil (f : C) {g h : C} (α : g ≅ h) :
    (whiskerLeftIso f α).hom = (whiskerLeftIso f α).hom := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.MonoidalCategory C
    f g h : C
    α : CategoryTheory.Iso g h
    ⊢ Eq (CategoryTheory.MonoidalCategory.whiskerLeftIso f α).hom (CategoryTheory. …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem evalWhiskerLeft_of_cons {f g h i j : C}
    (α : g ≅ h) (η : h ⟶ i) {ηs : i ⟶ j} {θ : f ⊗ i ⟶ f ⊗ j} (e_θ : f ◁ ηs = θ) :
    f ◁ (α.hom ≫ η ≫ ηs) = (whiskerLeftIso f α).hom ≫ f ◁ η ≫ θ := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.MonoidalCategory C
    f g h i j : C
    α : CategoryTheory.Iso g h
    η : Quiver.Hom h i
    ηs : Quiver.Hom i j
    θ : Quiver.Hom (CategoryTheory.MonoidalCategoryStruct.tensorObj f i) (Category …
    e_θ : Eq (CategoryTheory.MonoidalCategoryStruct.whiskerLeft f ηs) θ
    ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.whiskerLeft f (CategoryTheory.Cate …
  -/
  simp [e_θ]
  /-
    🎉 no goals
  -/


theorem evalWhiskerLeft_comp {f g h i : C}
    {η : h ⟶ i} {η₁ : g ⊗ h ⟶ g ⊗ i} {η₂ : f ⊗ g ⊗ h ⟶ f ⊗ g ⊗ i}
    {η₃ : f ⊗ g ⊗ h ⟶ (f ⊗ g) ⊗ i} {η₄ : (f ⊗ g) ⊗ h ⟶ (f ⊗ g) ⊗ i}
    (e_η₁ : g ◁ η = η₁) (e_η₂ : f ◁ η₁ = η₂)
    (e_η₃ : η₂ ≫ (α_ _ _ _).inv = η₃) (e_η₄ : (α_ _ _ _).hom ≫ η₃ = η₄) :
    (f ⊗ g) ◁ η = η₄ := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.MonoidalCategory C
    f g h i : C
    η : Quiver.Hom h i
    η₁ : Quiver.Hom (CategoryTheory.MonoidalCategoryStruct.tensorObj g h) (Categor …
    η₂ : Quiver.Hom (CategoryTheory.MonoidalCategoryStruct.tensorObj f (CategoryTh …
    η₃ : Quiver.Hom (CategoryTheory.MonoidalCategoryStruct.tensorObj f (CategoryTh …
    η₄ : Quiver.Hom (CategoryTheory.MonoidalCategoryStruct.tensorObj (CategoryTheo …
    e_η₁ : Eq (CategoryTheory.MonoidalCategoryStruct.whiskerLeft g η) η₁
    e_η₂ : Eq (CategoryTheory.MonoidalCategoryStruct.whiskerLeft f η₁) η₂
    e_η₃ : Eq (CategoryTheory.CategoryStruct.comp η₂ (CategoryTheory.MonoidalCateg …
    e_η₄ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategory …
    ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.whiskerLeft (CategoryTheory.Monoid …
  -/
  simp [e_η₁, e_η₂, e_η₃, e_η₄]
  /-
    🎉 no goals
  -/


theorem evalWhiskerLeft_id {f g : C} {η : f ⟶ g}
    {η₁ : f ⟶ 𝟙_ C ⊗ g} {η₂ : 𝟙_ C ⊗ f ⟶ 𝟙_ C ⊗ g}
    (e_η₁ : η ≫ (λ_ _).inv = η₁) (e_η₂ : (λ_ _).hom ≫ η₁ = η₂) :
    𝟙_ C ◁ η = η₂ := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.MonoidalCategory C
    f g : C
    η : Quiver.Hom f g
    η₁ : Quiver.Hom f (CategoryTheory.MonoidalCategoryStruct.tensorObj CategoryThe …
    η₂ : Quiver.Hom (CategoryTheory.MonoidalCategoryStruct.tensorObj CategoryTheor …
    e_η₁ : Eq (CategoryTheory.CategoryStruct.comp η (CategoryTheory.MonoidalCatego …
    e_η₂ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategory …
    ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.whiskerLeft CategoryTheory.Monoida …
  -/
  simp [e_η₁, e_η₂]
  /-
    🎉 no goals
  -/


theorem eval_whiskerLeft {f g h : C}
    {η η' : g ⟶ h} {θ : f ⊗ g ⟶ f ⊗ h}
    (e_η : η = η') (e_θ : f ◁ η' = θ) :
    f ◁ η = θ := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.MonoidalCategory C
    f g h : C
    η η' : Quiver.Hom g h
    θ : Quiver.Hom (CategoryTheory.MonoidalCategoryStruct.tensorObj f g) (Category …
    e_η : Eq η η'
    e_θ : Eq (CategoryTheory.MonoidalCategoryStruct.whiskerLeft f η') θ
    ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.whiskerLeft f η) θ
  -/
  simp [e_η, e_θ]
  /-
    🎉 no goals
  -/


theorem eval_whiskerRight {f g h : C}
    {η η' : f ⟶ g} {θ : f ⊗ h ⟶ g ⊗ h}
    (e_η : η = η') (e_θ : η' ▷ h = θ) :
    η ▷ h = θ := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.MonoidalCategory C
    f g h : C
    η η' : Quiver.Hom f g
    θ : Quiver.Hom (CategoryTheory.MonoidalCategoryStruct.tensorObj f h) (Category …
    e_η : Eq η η'
    e_θ : Eq (CategoryTheory.MonoidalCategoryStruct.whiskerRight η' h) θ
    ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.whiskerRight η h) θ
  -/
  simp [e_η, e_θ]
  /-
    🎉 no goals
  -/


theorem eval_tensorHom {f g h i : C}
    {η η' : f ⟶ g} {θ θ' : h ⟶ i} {ι : f ⊗ h ⟶ g ⊗ i}
    (e_η : η = η') (e_θ : θ = θ') (e_ι : η' ⊗ θ' = ι) :
    η ⊗ θ = ι := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.MonoidalCategory C
    f g h i : C
    η η' : Quiver.Hom f g
    θ θ' : Quiver.Hom h i
    ι : Quiver.Hom (CategoryTheory.MonoidalCategoryStruct.tensorObj f h) (Category …
    e_η : Eq η η'
    e_θ : Eq θ θ'
    e_ι : Eq (CategoryTheory.MonoidalCategoryStruct.tensorHom η' θ') ι
    ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.tensorHom η θ) ι
  -/
  simp [e_η, e_θ, e_ι]
  /-
    🎉 no goals
  -/


@[nolint synTaut]
theorem evalWhiskerRight_nil {f g : C} (α : f ≅ g) (h : C) :
    (whiskerRightIso α h).hom = (whiskerRightIso α h).hom := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.MonoidalCategory C
    f g : C
    α : CategoryTheory.Iso f g
    h : C
    ⊢ Eq (CategoryTheory.MonoidalCategory.whiskerRightIso α h).hom (CategoryTheory …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem evalWhiskerRight_cons_of_of {f g h i j : C}
    {α : f ≅ g} {η : g ⟶ h} {ηs : h ⟶ i} {ηs₁ : h ⊗ j ⟶ i ⊗ j}
    {η₁ : g ⊗ j ⟶ h ⊗ j} {η₂ : g ⊗ j ⟶ i ⊗ j} {η₃ : f ⊗ j ⟶ i ⊗ j}
    (e_ηs₁ : ηs ▷ j = ηs₁) (e_η₁ : η ▷ j = η₁)
    (e_η₂ : η₁ ≫ ηs₁ = η₂) (e_η₃ : (whiskerRightIso α j).hom ≫ η₂ = η₃) :
    (α.hom ≫ η ≫ ηs) ▷ j = η₃ := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.MonoidalCategory C
    f g h i j : C
    α : CategoryTheory.Iso f g
    η : Quiver.Hom g h
    ηs : Quiver.Hom h i
    ηs₁ : Quiver.Hom (CategoryTheory.MonoidalCategoryStruct.tensorObj h j) (Catego …
    η₁ : Quiver.Hom (CategoryTheory.MonoidalCategoryStruct.tensorObj g j) (Categor …
    η₂ : Quiver.Hom (CategoryTheory.MonoidalCategoryStruct.tensorObj g j) (Categor …
    η₃ : Quiver.Hom (CategoryTheory.MonoidalCategoryStruct.tensorObj f j) (Categor …
    e_ηs₁ : Eq (CategoryTheory.MonoidalCategoryStruct.whiskerRight ηs j) ηs₁
    e_η₁ : Eq (CategoryTheory.MonoidalCategoryStruct.whiskerRight η j) η₁
    e_η₂ : Eq (CategoryTheory.CategoryStruct.comp η₁ ηs₁) η₂
    e_η₃ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategory …
    ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.whiskerRight (CategoryTheory.Categ …
  -/
  simp_all
  /-
    🎉 no goals
  -/


theorem evalWhiskerRight_cons_whisker {f g h i j k : C}
    {α : g ≅ f ⊗ h} {η : h ⟶ i} {ηs : f ⊗ i ⟶ j}
    {η₁ : h ⊗ k ⟶ i ⊗ k} {η₂ : f ⊗ (h ⊗ k) ⟶ f ⊗ (i ⊗ k)} {ηs₁ : (f ⊗ i) ⊗ k ⟶ j ⊗ k}
    {ηs₂ : f ⊗ (i ⊗ k) ⟶ j ⊗ k} {η₃ : f ⊗ (h ⊗ k) ⟶ j ⊗ k} {η₄ : (f ⊗ h) ⊗ k ⟶ j ⊗ k}
    {η₅ : g ⊗ k ⟶ j ⊗ k}
    (e_η₁ : ((Iso.refl _).hom ≫ η ≫ (Iso.refl _).hom) ▷ k = η₁) (e_η₂ : f ◁ η₁ = η₂)
    (e_ηs₁ : ηs ▷ k = ηs₁) (e_ηs₂ : (α_ _ _ _).inv ≫ ηs₁ = ηs₂)
    (e_η₃ : η₂ ≫ ηs₂ = η₃) (e_η₄ : (α_ _ _ _).hom ≫ η₃ = η₄)
    (e_η₅ : (whiskerRightIso α k).hom ≫ η₄ = η₅) :
    (α.hom ≫ (f ◁ η) ≫ ηs) ▷ k = η₅ := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.MonoidalCategory C
    f g h i j k : C
    α : CategoryTheory.Iso g (CategoryTheory.MonoidalCategoryStruct.tensorObj f h)
    η : Quiver.Hom h i
    ηs : Quiver.Hom (CategoryTheory.MonoidalCategoryStruct.tensorObj f i) j
    η₁ : Quiver.Hom (CategoryTheory.MonoidalCategoryStruct.tensorObj h k) (Categor …
    η₂ : Quiver.Hom (CategoryTheory.MonoidalCategoryStruct.tensorObj f (CategoryTh …
    ηs₁ : Quiver.Hom (CategoryTheory.MonoidalCategoryStruct.tensorObj (CategoryThe …
    ηs₂ : Quiver.Hom (CategoryTheory.MonoidalCategoryStruct.tensorObj f (CategoryT …
    η₃ : Quiver.Hom (CategoryTheory.MonoidalCategoryStruct.tensorObj f (CategoryTh …
    η₄ : Quiver.Hom (CategoryTheory.MonoidalCategoryStruct.tensorObj (CategoryTheo …
    η₅ : Quiver.Hom (CategoryTheory.MonoidalCategoryStruct.tensorObj g k) (Categor …
    e_η₁ : Eq (CategoryTheory.MonoidalCategoryStruct.whiskerRight (CategoryTheory. …
    e_η₂ : Eq (CategoryTheory.MonoidalCategoryStruct.whiskerLeft f η₁) η₂
    e_ηs₁ : Eq (CategoryTheory.MonoidalCategoryStruct.whiskerRight ηs k) ηs₁
    e_ηs₂ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategor …
    e_η₃ : Eq (CategoryTheory.CategoryStruct.comp η₂ ηs₂) η₃
    e_η₄ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategory …
    e_η₅ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategory …
    ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.whiskerRight (CategoryTheory.Categ …
  -/
  simp at e_η₁ e_η₅
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.MonoidalCategory C
    f g h i j k : C
    α : CategoryTheory.Iso g (CategoryTheory.MonoidalCategoryStruct.tensorObj f h)
    η : Quiver.Hom h i
    ηs : Quiver.Hom (CategoryTheory.MonoidalCategoryStruct.tensorObj f i) j
    η₁ : Quiver.Hom (CategoryTheory.MonoidalCategoryStruct.tensorObj h k) (Categor …
    η₂ : Quiver.Hom (CategoryTheory.MonoidalCategoryStruct.tensorObj f (CategoryTh …
    ηs₁ : Quiver.Hom (CategoryTheory.MonoidalCategoryStruct.tensorObj (CategoryThe …
    ηs₂ : Quiver.Hom (CategoryTheory.MonoidalCategoryStruct.tensorObj f (CategoryT …
    η₃ : Quiver.Hom (CategoryTheory.MonoidalCategoryStruct.tensorObj f (CategoryTh …
    η₄ : Quiver.Hom (CategoryTheory.MonoidalCategoryStruct.tensorObj (CategoryTheo …
    η₅ : Quiver.Hom (CategoryTheory.MonoidalCategoryStruct.tensorObj g k) (Categor …
    e_η₂ : Eq (CategoryTheory.MonoidalCategoryStruct.whiskerLeft f η₁) η₂
    e_ηs₁ : Eq (CategoryTheory.MonoidalCategoryStruct.whiskerRight ηs k) ηs₁
    e_ηs₂ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategor …
    e_η₃ : Eq (CategoryTheory.CategoryStruct.comp η₂ ηs₂) η₃
    e_η₄ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategory …
    e_η₅ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategory …
    e_η₁ : Eq (CategoryTheory.MonoidalCategoryStruct.whiskerRight η k) η₁
    ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.whiskerRight (CategoryTheory.Categ …
  -/
  simp [e_η₁, e_η₂, e_ηs₁, e_ηs₂, e_η₃, e_η₄, e_η₅]
  /-
    🎉 no goals
  -/


theorem evalWhiskerRight_comp {f f' g h : C}
    {η : f ⟶ f'} {η₁ : f ⊗ g ⟶ f' ⊗ g} {η₂ : (f ⊗ g) ⊗ h ⟶ (f' ⊗ g) ⊗ h}
    {η₃ : (f ⊗ g) ⊗ h ⟶ f' ⊗ (g ⊗ h)} {η₄ : f ⊗ (g ⊗ h) ⟶ f' ⊗ (g ⊗ h)}
    (e_η₁ : η ▷ g = η₁) (e_η₂ : η₁ ▷ h = η₂)
    (e_η₃ : η₂ ≫ (α_ _ _ _).hom = η₃) (e_η₄ : (α_ _ _ _).inv ≫ η₃ = η₄) :
    η ▷ (g ⊗ h) = η₄ := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.MonoidalCategory C
    f f' g h : C
    η : Quiver.Hom f f'
    η₁ : Quiver.Hom (CategoryTheory.MonoidalCategoryStruct.tensorObj f g) (Categor …
    η₂ : Quiver.Hom (CategoryTheory.MonoidalCategoryStruct.tensorObj (CategoryTheo …
    η₃ : Quiver.Hom (CategoryTheory.MonoidalCategoryStruct.tensorObj (CategoryTheo …
    η₄ : Quiver.Hom (CategoryTheory.MonoidalCategoryStruct.tensorObj f (CategoryTh …
    e_η₁ : Eq (CategoryTheory.MonoidalCategoryStruct.whiskerRight η g) η₁
    e_η₂ : Eq (CategoryTheory.MonoidalCategoryStruct.whiskerRight η₁ h) η₂
    e_η₃ : Eq (CategoryTheory.CategoryStruct.comp η₂ (CategoryTheory.MonoidalCateg …
    e_η₄ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategory …
    ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.whiskerRight η (CategoryTheory.Mon …
  -/
  simp [e_η₁, e_η₂, e_η₃, e_η₄]
  /-
    🎉 no goals
  -/


theorem evalWhiskerRight_id {f g : C}
    {η : f ⟶ g} {η₁ : f ⟶ g ⊗ 𝟙_ C} {η₂ : f ⊗ 𝟙_ C ⟶ g ⊗ 𝟙_ C}
    (e_η₁ : η ≫ (ρ_ _).inv = η₁) (e_η₂ : (ρ_ _).hom ≫ η₁ = η₂) :
    η ▷ 𝟙_ C = η₂ := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.MonoidalCategory C
    f g : C
    η : Quiver.Hom f g
    η₁ : Quiver.Hom f (CategoryTheory.MonoidalCategoryStruct.tensorObj g CategoryT …
    η₂ : Quiver.Hom (CategoryTheory.MonoidalCategoryStruct.tensorObj f CategoryThe …
    e_η₁ : Eq (CategoryTheory.CategoryStruct.comp η (CategoryTheory.MonoidalCatego …
    e_η₂ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategory …
    ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.whiskerRight η CategoryTheory.Mono …
  -/
  simp [e_η₁, e_η₂]
  /-
    🎉 no goals
  -/


theorem evalWhiskerRightAux_of {f g : C} (η : f ⟶ g) (h : C) :
    η ▷ h = (Iso.refl _).hom ≫ η ▷ h ≫ (Iso.refl _).hom := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.MonoidalCategory C
    f g : C
    η : Quiver.Hom f g
    h : C
    ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.whiskerRight η h) (CategoryTheory. …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem evalWhiskerRightAux_cons {f g h i j : C} {η : g ⟶ h} {ηs : i ⟶ j}
    {ηs' : i ⊗ f ⟶ j ⊗ f} {η₁ : g ⊗ (i ⊗ f) ⟶ h ⊗ (j ⊗ f)}
    {η₂ : g ⊗ (i ⊗ f) ⟶ (h ⊗ j) ⊗ f} {η₃ : (g ⊗ i) ⊗ f ⟶ (h ⊗ j) ⊗ f}
    (e_ηs' : ηs ▷ f = ηs') (e_η₁ : ((Iso.refl _).hom ≫ η ≫ (Iso.refl _).hom) ⊗ ηs' = η₁)
    (e_η₂ : η₁ ≫ (α_ _ _ _).inv = η₂) (e_η₃ : (α_ _ _ _).hom ≫ η₂ = η₃) :
    (η ⊗ ηs) ▷ f = η₃ := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.MonoidalCategory C
    f g h i j : C
    η : Quiver.Hom g h
    ηs : Quiver.Hom i j
    ηs' : Quiver.Hom (CategoryTheory.MonoidalCategoryStruct.tensorObj i f) (Catego …
    η₁ : Quiver.Hom (CategoryTheory.MonoidalCategoryStruct.tensorObj g (CategoryTh …
    η₂ : Quiver.Hom (CategoryTheory.MonoidalCategoryStruct.tensorObj g (CategoryTh …
    η₃ : Quiver.Hom (CategoryTheory.MonoidalCategoryStruct.tensorObj (CategoryTheo …
    e_ηs' : Eq (CategoryTheory.MonoidalCategoryStruct.whiskerRight ηs f) ηs'
    e_η₁ : Eq (CategoryTheory.MonoidalCategoryStruct.tensorHom (CategoryTheory.Cat …
    e_η₂ : Eq (CategoryTheory.CategoryStruct.comp η₁ (CategoryTheory.MonoidalCateg …
    e_η₃ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategory …
    ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.whiskerRight (CategoryTheory.Monoi …
  -/
  simp [← e_ηs', ← e_η₁, ← e_η₂, ← e_η₃, MonoidalCategory.tensorHom_def]
  /-
    🎉 no goals
  -/


theorem evalWhiskerRight_cons_of {f f' g h i : C} {α : f' ≅ g} {η : g ⟶ h} {ηs : h ⟶ i}
    {ηs₁ : h ⊗ f ⟶ i ⊗ f} {η₁ : g ⊗ f ⟶ h ⊗ f} {η₂ : g ⊗ f ⟶ i ⊗ f}
    {η₃ : f' ⊗ f ⟶ i ⊗ f}
    (e_ηs₁ : ηs ▷ f = ηs₁) (e_η₁ : η ▷ f = η₁)
    (e_η₂ : η₁ ≫ ηs₁ = η₂) (e_η₃ : (whiskerRightIso α f).hom ≫ η₂ = η₃) :
    (α.hom ≫ η ≫ ηs) ▷ f = η₃ := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.MonoidalCategory C
    f f' g h i : C
    α : CategoryTheory.Iso f' g
    η : Quiver.Hom g h
    ηs : Quiver.Hom h i
    ηs₁ : Quiver.Hom (CategoryTheory.MonoidalCategoryStruct.tensorObj h f) (Catego …
    η₁ : Quiver.Hom (CategoryTheory.MonoidalCategoryStruct.tensorObj g f) (Categor …
    η₂ : Quiver.Hom (CategoryTheory.MonoidalCategoryStruct.tensorObj g f) (Categor …
    η₃ : Quiver.Hom (CategoryTheory.MonoidalCategoryStruct.tensorObj f' f) (Catego …
    e_ηs₁ : Eq (CategoryTheory.MonoidalCategoryStruct.whiskerRight ηs f) ηs₁
    e_η₁ : Eq (CategoryTheory.MonoidalCategoryStruct.whiskerRight η f) η₁
    e_η₂ : Eq (CategoryTheory.CategoryStruct.comp η₁ ηs₁) η₂
    e_η₃ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategory …
    ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.whiskerRight (CategoryTheory.Categ …
  -/
  simp_all
  /-
    🎉 no goals
  -/


theorem evalHorizontalCompAux_of {f g h i : C} (η : f ⟶ g) (θ : h ⟶ i) :
    η ⊗ θ = (Iso.refl _).hom ≫ (η ⊗ θ) ≫ (Iso.refl _).hom := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.MonoidalCategory C
    f g h i : C
    η : Quiver.Hom f g
    θ : Quiver.Hom h i
    ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.tensorHom η θ) (CategoryTheory.Cat …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem evalHorizontalCompAux_cons {f f' g g' h i : C} {η : f ⟶ g} {ηs : f' ⟶ g'} {θ : h ⟶ i}
    {ηθ : f' ⊗ h ⟶ g' ⊗ i} {η₁ : f ⊗ (f' ⊗ h) ⟶ g ⊗ (g' ⊗ i)}
    {ηθ₁ : f ⊗ (f' ⊗ h) ⟶ (g ⊗ g') ⊗ i} {ηθ₂ : (f ⊗ f') ⊗ h ⟶ (g ⊗ g') ⊗ i}
    (e_ηθ : ηs ⊗ θ = ηθ) (e_η₁ : ((Iso.refl _).hom ≫ η ≫ (Iso.refl _).hom) ⊗ ηθ = η₁)
    (e_ηθ₁ : η₁ ≫ (α_ _ _ _).inv = ηθ₁) (e_ηθ₂ : (α_ _ _ _).hom ≫ ηθ₁ = ηθ₂) :
    (η ⊗ ηs) ⊗ θ = ηθ₂ := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.MonoidalCategory C
    f f' g g' h i : C
    η : Quiver.Hom f g
    ηs : Quiver.Hom f' g'
    θ : Quiver.Hom h i
    ηθ : Quiver.Hom (CategoryTheory.MonoidalCategoryStruct.tensorObj f' h) (Catego …
    η₁ : Quiver.Hom (CategoryTheory.MonoidalCategoryStruct.tensorObj f (CategoryTh …
    ηθ₁ : Quiver.Hom (CategoryTheory.MonoidalCategoryStruct.tensorObj f (CategoryT …
    ηθ₂ : Quiver.Hom (CategoryTheory.MonoidalCategoryStruct.tensorObj (CategoryThe …
    e_ηθ : Eq (CategoryTheory.MonoidalCategoryStruct.tensorHom ηs θ) ηθ
    e_η₁ : Eq (CategoryTheory.MonoidalCategoryStruct.tensorHom (CategoryTheory.Cat …
    e_ηθ₁ : Eq (CategoryTheory.CategoryStruct.comp η₁ (CategoryTheory.MonoidalCate …
    e_ηθ₂ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategor …
    ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.tensorHom (CategoryTheory.Monoidal …
  -/
  simp_all
  /-
    🎉 no goals
  -/


theorem evalHorizontalCompAux'_whisker {f f' g g' h : C} {η : g ⟶ h} {θ : f' ⟶ g'}
    {ηθ : g ⊗ f' ⟶ h ⊗ g'} {η₁ : f ⊗ (g ⊗ f') ⟶ f ⊗ (h ⊗ g')}
    {η₂ :  f ⊗ (g ⊗ f') ⟶ (f ⊗ h) ⊗ g'} {η₃ : (f ⊗ g) ⊗ f' ⟶ (f ⊗ h) ⊗ g'}
    (e_ηθ : η ⊗ θ = ηθ) (e_η₁ : f ◁ ηθ = η₁)
    (e_η₂ : η₁ ≫ (α_ _ _ _).inv = η₂) (e_η₃ : (α_ _ _ _).hom ≫ η₂ = η₃) :
    (f ◁ η) ⊗ θ = η₃ := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.MonoidalCategory C
    f f' g g' h : C
    η : Quiver.Hom g h
    θ : Quiver.Hom f' g'
    ηθ : Quiver.Hom (CategoryTheory.MonoidalCategoryStruct.tensorObj g f') (Catego …
    η₁ : Quiver.Hom (CategoryTheory.MonoidalCategoryStruct.tensorObj f (CategoryTh …
    η₂ : Quiver.Hom (CategoryTheory.MonoidalCategoryStruct.tensorObj f (CategoryTh …
    η₃ : Quiver.Hom (CategoryTheory.MonoidalCategoryStruct.tensorObj (CategoryTheo …
    e_ηθ : Eq (CategoryTheory.MonoidalCategoryStruct.tensorHom η θ) ηθ
    e_η₁ : Eq (CategoryTheory.MonoidalCategoryStruct.whiskerLeft f ηθ) η₁
    e_η₂ : Eq (CategoryTheory.CategoryStruct.comp η₁ (CategoryTheory.MonoidalCateg …
    e_η₃ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategory …
    ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.tensorHom (CategoryTheory.Monoidal …
  -/
  simp only [← e_ηθ, ← e_η₁, ← e_η₂, ← e_η₃]
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.MonoidalCategory C
    f f' g g' h : C
    η : Quiver.Hom g h
    θ : Quiver.Hom f' g'
    ηθ : Quiver.Hom (CategoryTheory.MonoidalCategoryStruct.tensorObj g f') (Catego …
    η₁ : Quiver.Hom (CategoryTheory.MonoidalCategoryStruct.tensorObj f (CategoryTh …
    η₂ : Quiver.Hom (CategoryTheory.MonoidalCategoryStruct.tensorObj f (CategoryTh …
    η₃ : Quiver.Hom (CategoryTheory.MonoidalCategoryStruct.tensorObj (CategoryTheo …
    e_ηθ : Eq (CategoryTheory.MonoidalCategoryStruct.tensorHom η θ) ηθ
    e_η₁ : Eq (CategoryTheory.MonoidalCategoryStruct.whiskerLeft f ηθ) η₁
    e_η₂ : Eq (CategoryTheory.CategoryStruct.comp η₁ (CategoryTheory.MonoidalCateg …
    e_η₃ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategory …
    ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.tensorHom (CategoryTheory.Monoidal …
  -/
  simp [MonoidalCategory.tensorHom_def]
  /-
    🎉 no goals
  -/


theorem evalHorizontalCompAux'_of_whisker {f f' g g' h : C} {η : g ⟶ h} {θ : f' ⟶ g'}
    {η₁ : g ⊗ f ⟶ h ⊗ f} {ηθ : (g ⊗ f) ⊗ f' ⟶ (h ⊗ f) ⊗ g'}
    {ηθ₁ : (g ⊗ f) ⊗ f' ⟶ h ⊗ (f ⊗ g')}
    {ηθ₂ : g ⊗ (f ⊗ f') ⟶ h ⊗ (f ⊗ g')}
    (e_η₁ : η ▷ f = η₁) (e_ηθ : η₁ ⊗ ((Iso.refl _).hom ≫ θ ≫ (Iso.refl _).hom) = ηθ)
    (e_ηθ₁ : ηθ ≫ (α_ _ _ _).hom = ηθ₁) (e_ηθ₂ : (α_ _ _ _).inv ≫ ηθ₁ = ηθ₂) :
    η ⊗ (f ◁ θ) = ηθ₂ := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.MonoidalCategory C
    f f' g g' h : C
    η : Quiver.Hom g h
    θ : Quiver.Hom f' g'
    η₁ : Quiver.Hom (CategoryTheory.MonoidalCategoryStruct.tensorObj g f) (Categor …
    ηθ : Quiver.Hom (CategoryTheory.MonoidalCategoryStruct.tensorObj (CategoryTheo …
    ηθ₁ : Quiver.Hom (CategoryTheory.MonoidalCategoryStruct.tensorObj (CategoryThe …
    ηθ₂ : Quiver.Hom (CategoryTheory.MonoidalCategoryStruct.tensorObj g (CategoryT …
    e_η₁ : Eq (CategoryTheory.MonoidalCategoryStruct.whiskerRight η f) η₁
    e_ηθ : Eq (CategoryTheory.MonoidalCategoryStruct.tensorHom η₁ (CategoryTheory. …
    e_ηθ₁ : Eq (CategoryTheory.CategoryStruct.comp ηθ (CategoryTheory.MonoidalCate …
    e_ηθ₂ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategor …
    ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.tensorHom η (CategoryTheory.Monoid …
  -/
  simp only [← e_η₁, ← e_ηθ, ← e_ηθ₁, ← e_ηθ₂]
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.MonoidalCategory C
    f f' g g' h : C
    η : Quiver.Hom g h
    θ : Quiver.Hom f' g'
    η₁ : Quiver.Hom (CategoryTheory.MonoidalCategoryStruct.tensorObj g f) (Categor …
    ηθ : Quiver.Hom (CategoryTheory.MonoidalCategoryStruct.tensorObj (CategoryTheo …
    ηθ₁ : Quiver.Hom (CategoryTheory.MonoidalCategoryStruct.tensorObj (CategoryThe …
    ηθ₂ : Quiver.Hom (CategoryTheory.MonoidalCategoryStruct.tensorObj g (CategoryT …
    e_η₁ : Eq (CategoryTheory.MonoidalCategoryStruct.whiskerRight η f) η₁
    e_ηθ : Eq (CategoryTheory.MonoidalCategoryStruct.tensorHom η₁ (CategoryTheory. …
    e_ηθ₁ : Eq (CategoryTheory.CategoryStruct.comp ηθ (CategoryTheory.MonoidalCate …
    e_ηθ₂ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategor …
    ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.tensorHom η (CategoryTheory.Monoid …
  -/
  simp [MonoidalCategory.tensorHom_def]
  /-
    🎉 no goals
  -/


@[nolint synTaut]
theorem evalHorizontalComp_nil_nil {f g h i : C} (α : f ≅ g) (β : h ≅ i) :
    (α ⊗ β).hom = (α ⊗ β).hom := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.MonoidalCategory C
    f g h i : C
    α : CategoryTheory.Iso f g
    β : CategoryTheory.Iso h i
    ⊢ Eq (CategoryTheory.MonoidalCategory.tensorIso α β).hom (CategoryTheory.Monoi …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem evalHorizontalComp_nil_cons {f f' g g' h i : C}
    {α : f ≅ g} {β : f' ≅ g'} {η : g' ⟶ h} {ηs : h ⟶ i}
    {η₁ : g ⊗ g' ⟶ g ⊗ h} {ηs₁ : g ⊗ h ⟶ g ⊗ i}
    {η₂ : g ⊗ g' ⟶ g ⊗ i} {η₃ : f ⊗ f' ⟶ g ⊗ i}
    (e_η₁ : g ◁ ((Iso.refl _).hom ≫ η ≫ (Iso.refl _).hom) = η₁)
    (e_ηs₁ : g ◁ ηs = ηs₁) (e_η₂ : η₁ ≫ ηs₁ = η₂)
    (e_η₃ : (α ⊗ β).hom ≫ η₂ = η₃) :
    α.hom ⊗ (β.hom ≫ η ≫ ηs) = η₃ := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.MonoidalCategory C
    f f' g g' h i : C
    α : CategoryTheory.Iso f g
    β : CategoryTheory.Iso f' g'
    η : Quiver.Hom g' h
    ηs : Quiver.Hom h i
    η₁ : Quiver.Hom (CategoryTheory.MonoidalCategoryStruct.tensorObj g g') (Catego …
    ηs₁ : Quiver.Hom (CategoryTheory.MonoidalCategoryStruct.tensorObj g h) (Catego …
    η₂ : Quiver.Hom (CategoryTheory.MonoidalCategoryStruct.tensorObj g g') (Catego …
    η₃ : Quiver.Hom (CategoryTheory.MonoidalCategoryStruct.tensorObj f f') (Catego …
    e_η₁ : Eq (CategoryTheory.MonoidalCategoryStruct.whiskerLeft g (CategoryTheory …
    e_ηs₁ : Eq (CategoryTheory.MonoidalCategoryStruct.whiskerLeft g ηs) ηs₁
    e_η₂ : Eq (CategoryTheory.CategoryStruct.comp η₁ ηs₁) η₂
    e_η₃ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategory …
    ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.tensorHom α.hom (CategoryTheory.Ca …
  -/
  simp_all [MonoidalCategory.tensorHom_def]
  /-
    🎉 no goals
  -/


theorem evalHorizontalComp_cons_nil {f f' g g' h i : C}
    {α : f ≅ g} {η : g ⟶ h} {ηs : h ⟶ i} {β : f' ≅ g'}
    {η₁ : g ⊗ g' ⟶ h ⊗ g'} {ηs₁ : h ⊗ g' ⟶ i ⊗ g'} {η₂ : g ⊗ g' ⟶ i ⊗ g'} {η₃ : f ⊗ f' ⟶ i ⊗ g'}
    (e_η₁ : ((Iso.refl _).hom ≫ η ≫ (Iso.refl _).hom) ▷ g' = η₁) (e_ηs₁ : ηs ▷ g' = ηs₁)
    (e_η₂ : η₁ ≫ ηs₁ = η₂) (e_η₃ : (α ⊗ β).hom ≫ η₂ = η₃) :
    (α.hom ≫ η ≫ ηs) ⊗ β.hom = η₃ := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.MonoidalCategory C
    f f' g g' h i : C
    α : CategoryTheory.Iso f g
    η : Quiver.Hom g h
    ηs : Quiver.Hom h i
    β : CategoryTheory.Iso f' g'
    η₁ : Quiver.Hom (CategoryTheory.MonoidalCategoryStruct.tensorObj g g') (Catego …
    ηs₁ : Quiver.Hom (CategoryTheory.MonoidalCategoryStruct.tensorObj h g') (Categ …
    η₂ : Quiver.Hom (CategoryTheory.MonoidalCategoryStruct.tensorObj g g') (Catego …
    η₃ : Quiver.Hom (CategoryTheory.MonoidalCategoryStruct.tensorObj f f') (Catego …
    e_η₁ : Eq (CategoryTheory.MonoidalCategoryStruct.whiskerRight (CategoryTheory. …
    e_ηs₁ : Eq (CategoryTheory.MonoidalCategoryStruct.whiskerRight ηs g') ηs₁
    e_η₂ : Eq (CategoryTheory.CategoryStruct.comp η₁ ηs₁) η₂
    e_η₃ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategory …
    ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.tensorHom (CategoryTheory.Category …
  -/
  simp_all [MonoidalCategory.tensorHom_def']
  /-
    🎉 no goals
  -/


theorem evalHorizontalComp_cons_cons {f f' g g' h h' i i' : C}
    {α : f ≅ g} {η : g ⟶ h} {ηs : h ⟶ i}
    {β : f' ≅ g'} {θ : g' ⟶ h'} {θs : h' ⟶ i'}
    {ηθ : g ⊗ g' ⟶ h ⊗ h'} {ηθs : h ⊗ h' ⟶ i ⊗ i'}
    {ηθ₁ : g ⊗ g' ⟶ i ⊗ i'} {ηθ₂ : f ⊗ f' ⟶ i ⊗ i'}
    (e_ηθ : η ⊗ θ = ηθ) (e_ηθs : ηs ⊗ θs = ηθs)
    (e_ηθ₁ : ηθ ≫ ηθs = ηθ₁) (e_ηθ₂ : (α ⊗ β).hom ≫ ηθ₁ = ηθ₂) :
    (α.hom ≫ η ≫ ηs) ⊗ (β.hom ≫ θ ≫ θs) = ηθ₂ := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.MonoidalCategory C
    f f' g g' h h' i i' : C
    α : CategoryTheory.Iso f g
    η : Quiver.Hom g h
    ηs : Quiver.Hom h i
    β : CategoryTheory.Iso f' g'
    θ : Quiver.Hom g' h'
    θs : Quiver.Hom h' i'
    ηθ : Quiver.Hom (CategoryTheory.MonoidalCategoryStruct.tensorObj g g') (Catego …
    ηθs : Quiver.Hom (CategoryTheory.MonoidalCategoryStruct.tensorObj h h') (Categ …
    ηθ₁ : Quiver.Hom (CategoryTheory.MonoidalCategoryStruct.tensorObj g g') (Categ …
    ηθ₂ : Quiver.Hom (CategoryTheory.MonoidalCategoryStruct.tensorObj f f') (Categ …
    e_ηθ : Eq (CategoryTheory.MonoidalCategoryStruct.tensorHom η θ) ηθ
    e_ηθs : Eq (CategoryTheory.MonoidalCategoryStruct.tensorHom ηs θs) ηθs
    e_ηθ₁ : Eq (CategoryTheory.CategoryStruct.comp ηθ ηθs) ηθ₁
    e_ηθ₂ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategor …
    ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.tensorHom (CategoryTheory.Category …
  -/
  simp [← e_ηθ , ← e_ηθs , ← e_ηθ₁, ← e_ηθ₂]
  /-
    🎉 no goals
  -/


instance : MkEvalComp MonoidalM where
  mkEvalCompNilNil α β := do
    let ctx ← read
    let _cat := ctx.instCat
    let f ← α.srcM
    let g ← α.tgtM
    let h ← β.tgtM
    have f : Q($ctx.C) := f.e
    have g : Q($ctx.C) := g.e
    have h : Q($ctx.C) := h.e
    have α : Q($f ≅ $g) := α.e
    have β : Q($g ≅ $h) := β.e
    return q(evalComp_nil_nil $α $β)
  mkEvalCompNilCons α β η ηs := do
    let ctx ← read
    let _cat := ctx.instCat
    let f ← α.srcM
    let g ← α.tgtM
    let h ← β.tgtM
    let i ← η.tgtM
    let j ← ηs.tgtM
    have f : Q($ctx.C) := f.e
    have g : Q($ctx.C) := g.e
    have h : Q($ctx.C) := h.e
    have i : Q($ctx.C) := i.e
    have j : Q($ctx.C) := j.e
    have α : Q($f ≅ $g) := α.e
    have β : Q($g ≅ $h) := β.e
    have η : Q($h ⟶ $i) := η.e.e
    have ηs : Q($i ⟶ $j) := ηs.e.e
    return q(evalComp_nil_cons $α $β $η $ηs)
  mkEvalCompCons α η ηs θ ι e_ι := do
    let ctx ← read
    let _cat := ctx.instCat
    let f ← α.srcM
    let g ← α.tgtM
    let h ← η.tgtM
    let i ← ηs.tgtM
    let j ← θ.tgtM
    have f : Q($ctx.C) := f.e
    have g : Q($ctx.C) := g.e
    have h : Q($ctx.C) := h.e
    have i : Q($ctx.C) := i.e
    have j : Q($ctx.C) := j.e
    have α : Q($f ≅ $g) := α.e
    have η : Q($g ⟶ $h) := η.e.e
    have ηs : Q($h ⟶ $i) := ηs.e.e
    have θ : Q($i ⟶ $j) := θ.e.e
    have ι : Q($h ⟶ $j) := ι.e.e
    have e_ι : Q($ηs ≫ $θ = $ι) := e_ι
    return q(evalComp_cons $α $η $e_ι)


instance : MkEvalWhiskerLeft MonoidalM where
  mkEvalWhiskerLeftNil f α := do
    let ctx ← read
    let .some _monoidal := ctx.instMonoidal? | synthMonoidalError
    let g ← α.srcM
    let h ← α.tgtM
    have f_e : Q($ctx.C) := f.e
    have g_e : Q($ctx.C) := g.e
    have h_e : Q($ctx.C) := h.e
    have α_e : Q($g_e ≅ $h_e) := α.e
    return q(evalWhiskerLeft_nil $f_e $α_e)
  mkEvalWhiskerLeftOfCons f α η ηs θ e_θ := do
    let ctx ← read
    let .some _monoidal := ctx.instMonoidal? | synthMonoidalError
    let g ← α.srcM
    let h ← α.tgtM
    let i ← η.tgtM
    let j ← ηs.tgtM
    have f : Q($ctx.C) := f.e
    have g : Q($ctx.C) := g.e
    have h : Q($ctx.C) := h.e
    have i : Q($ctx.C) := i.e
    have j : Q($ctx.C) := j.e
    have α : Q($g ≅ $h) := α.e
    have η : Q($h ⟶ $i) := η.e.e
    have ηs : Q($i ⟶ $j) := ηs.e.e
    have θ : Q($f ⊗ $i ⟶ $f ⊗ $j) := θ.e.e
    have e_θ : Q($f ◁ $ηs = $θ) := e_θ
    return q(evalWhiskerLeft_of_cons $α $η $e_θ)
  mkEvalWhiskerLeftComp f g η η₁ η₂ η₃ η₄ e_η₁ e_η₂ e_η₃ e_η₄ := do
    let ctx ← read
    let .some _monoidal := ctx.instMonoidal? | synthMonoidalError
    let h ← η.srcM
    let i ← η.tgtM
    have f : Q($ctx.C) := f.e
    have g : Q($ctx.C) := g.e
    have h : Q($ctx.C) := h.e
    have i : Q($ctx.C) := i.e
    have η : Q($h ⟶ $i) := η.e.e
    have η₁ : Q($g ⊗ $h ⟶ $g ⊗ $i) := η₁.e.e
    have η₂ : Q($f ⊗ $g ⊗ $h ⟶ $f ⊗ $g ⊗ $i) := η₂.e.e
    have η₃ : Q($f ⊗ $g ⊗ $h ⟶ ($f ⊗ $g) ⊗ $i) := η₃.e.e
    have η₄ : Q(($f ⊗ $g) ⊗ $h ⟶ ($f ⊗ $g) ⊗ $i) := η₄.e.e
    have e_η₁ : Q($g ◁ $η = $η₁) := e_η₁
    have e_η₂ : Q($f ◁ $η₁ = $η₂) := e_η₂
    have e_η₃ : Q($η₂ ≫ (α_ _ _ _).inv = $η₃) := e_η₃
    have e_η₄ : Q((α_ _ _ _).hom ≫ $η₃ = $η₄) := e_η₄
    return q(evalWhiskerLeft_comp $e_η₁ $e_η₂ $e_η₃ $e_η₄)
  mkEvalWhiskerLeftId η η₁ η₂ e_η₁ e_η₂ := do
    let ctx ← read
    let .some _monoidal := ctx.instMonoidal? | synthMonoidalError
    let f ← η.srcM
    let g ← η.tgtM
    have f : Q($ctx.C) := f.e
    have g : Q($ctx.C) := g.e
    have η : Q($f ⟶ $g) := η.e.e
    have η₁ : Q($f ⟶ 𝟙_ _ ⊗ $g) := η₁.e.e
    have η₂ : Q(𝟙_ _ ⊗ $f ⟶ 𝟙_ _ ⊗ $g) := η₂.e.e
    have e_η₁ : Q($η ≫ (λ_ _).inv = $η₁) := e_η₁
    have e_η₂ : Q((λ_ _).hom ≫ $η₁ = $η₂) := e_η₂
    return q(evalWhiskerLeft_id $e_η₁ $e_η₂)


instance : MkEvalWhiskerRight MonoidalM where
  mkEvalWhiskerRightAuxOf η h := do
    let ctx ← read
    let .some _monoidal := ctx.instMonoidal? | synthMonoidalError
    let f ← η.srcM
    let g ← η.tgtM
    have f : Q($ctx.C) := f.e
    have g : Q($ctx.C) := g.e
    have η : Q($f ⟶ $g) := η.e.e
    have h : Q($ctx.C) := h.e
    return q(evalWhiskerRightAux_of $η $h)
  mkEvalWhiskerRightAuxCons f η ηs ηs' η₁ η₂ η₃ e_ηs' e_η₁ e_η₂ e_η₃ := do
    let ctx ← read
    let .some _monoidal := ctx.instMonoidal? | synthMonoidalError
    let g ← η.srcM
    let h ← η.tgtM
    let i ← ηs.srcM
    let j ← ηs.tgtM
    have f : Q($ctx.C) := f.e
    have g : Q($ctx.C) := g.e
    have h : Q($ctx.C) := h.e
    have i : Q($ctx.C) := i.e
    have j : Q($ctx.C) := j.e
    have η : Q($g ⟶ $h) := η.e.e
    have ηs : Q($i ⟶ $j) := ηs.e.e
    have ηs' : Q($i ⊗ $f ⟶ $j ⊗ $f) := ηs'.e.e
    have η₁ : Q($g ⊗ ($i ⊗ $f) ⟶ $h ⊗ ($j ⊗ $f)) := η₁.e.e
    have η₂ : Q($g ⊗ ($i ⊗ $f) ⟶ ($h ⊗ $j) ⊗ $f) := η₂.e.e
    have η₃ : Q(($g ⊗ $i) ⊗ $f ⟶ ($h ⊗ $j) ⊗ $f) := η₃.e.e
    have e_ηs' : Q($ηs ▷ $f = $ηs') := e_ηs'
    have e_η₁ : Q(((Iso.refl _).hom ≫ $η ≫ (Iso.refl _).hom) ⊗ $ηs' = $η₁) := e_η₁
    have e_η₂ : Q($η₁ ≫ (α_ _ _ _).inv = $η₂) := e_η₂
    have e_η₃ : Q((α_ _ _ _).hom ≫ $η₂ = $η₃) := e_η₃
    return q(evalWhiskerRightAux_cons $e_ηs' $e_η₁ $e_η₂ $e_η₃)
  mkEvalWhiskerRightNil α h := do
    let ctx ← read
    let .some _monoidal := ctx.instMonoidal? | synthMonoidalError
    let f ← α.srcM
    let g ← α.tgtM
    have f : Q($ctx.C) := f.e
    have g : Q($ctx.C) := g.e
    have h : Q($ctx.C) := h.e
    have α : Q($f ≅ $g) := α.e
    return q(evalWhiskerRight_nil $α $h)
  mkEvalWhiskerRightConsOfOf j α η ηs ηs₁ η₁ η₂ η₃ e_ηs₁ e_η₁ e_η₂ e_η₃ := do
    let ctx ← read
    let .some _monoidal := ctx.instMonoidal? | synthMonoidalError
    let f ← α.srcM
    let g ← α.tgtM
    let h ← η.tgtM
    let i ← ηs.tgtM
    have f : Q($ctx.C) := f.e
    have g : Q($ctx.C) := g.e
    have h : Q($ctx.C) := h.e
    have i : Q($ctx.C) := i.e
    have j : Q($ctx.C) := j.e
    have α : Q($f ≅ $g) := α.e
    have η : Q($g ⟶ $h) := η.e.e
    have ηs : Q($h ⟶ $i) := ηs.e.e
    have ηs₁ : Q($h ⊗ $j ⟶ $i ⊗ $j) := ηs₁.e.e
    have η₁ : Q($g ⊗ $j ⟶ $h ⊗ $j) := η₁.e.e
    have η₂ : Q($g ⊗ $j ⟶ $i ⊗ $j) := η₂.e.e
    have η₃ : Q($f ⊗ $j ⟶ $i ⊗ $j) := η₃.e.e
    have e_ηs₁ : Q($ηs ▷ $j = $ηs₁) := e_ηs₁
    have e_η₁ : Q($η ▷ $j = $η₁) := e_η₁
    have e_η₂ : Q($η₁ ≫ $ηs₁ = $η₂) := e_η₂
    have e_η₃ : Q((whiskerRightIso $α $j).hom ≫ $η₂ = $η₃) := e_η₃
    return q(evalWhiskerRight_cons_of_of $e_ηs₁ $e_η₁ $e_η₂ $e_η₃)
  mkEvalWhiskerRightConsWhisker f k α η ηs η₁ η₂ ηs₁ ηs₂ η₃ η₄ η₅
      e_η₁ e_η₂ e_ηs₁ e_ηs₂ e_η₃ e_η₄ e_η₅ := do
    let ctx ← read
    let .some _monoidal := ctx.instMonoidal? | synthMonoidalError
    let g ← α.srcM
    let h ← η.srcM
    let i ← η.tgtM
    let j ← ηs.tgtM
    have f : Q($ctx.C) := f.e
    have g : Q($ctx.C) := g.e
    have h : Q($ctx.C) := h.e
    have i : Q($ctx.C) := i.e
    have j : Q($ctx.C) := j.e
    have k : Q($ctx.C) := k.e
    have α : Q($g ≅ $f ⊗ $h) := α.e
    have η : Q($h ⟶ $i) := η.e.e
    have ηs : Q($f ⊗ $i ⟶ $j) := ηs.e.e
    have η₁ : Q($h ⊗ $k ⟶ $i ⊗ $k) := η₁.e.e
    have η₂ : Q($f ⊗ ($h ⊗ $k) ⟶ $f ⊗ ($i ⊗ $k)) := η₂.e.e
    have ηs₁ : Q(($f ⊗ $i) ⊗ $k ⟶ $j ⊗ $k) := ηs₁.e.e
    have ηs₂ : Q($f ⊗ ($i ⊗ $k) ⟶ $j ⊗ $k) := ηs₂.e.e
    have η₃ : Q($f ⊗ ($h ⊗ $k) ⟶ $j ⊗ $k) := η₃.e.e
    have η₄ : Q(($f ⊗ $h) ⊗ $k ⟶ $j ⊗ $k) := η₄.e.e
    have η₅ : Q($g ⊗ $k ⟶ $j ⊗ $k) := η₅.e.e
    have e_η₁ : Q(((Iso.refl _).hom ≫ $η ≫ (Iso.refl _).hom) ▷ $k = $η₁) := e_η₁
    have e_η₂ : Q($f ◁ $η₁ = $η₂) := e_η₂
    have e_ηs₁ : Q($ηs ▷ $k = $ηs₁) := e_ηs₁
    have e_ηs₂ : Q((α_ _ _ _).inv ≫ $ηs₁ = $ηs₂) := e_ηs₂
    have e_η₃ : Q($η₂ ≫ $ηs₂ = $η₃) := e_η₃
    have e_η₄ : Q((α_ _ _ _).hom ≫ $η₃ = $η₄) := e_η₄
    have e_η₅ : Q((whiskerRightIso $α $k).hom ≫ $η₄ = $η₅) := e_η₅
    return q(evalWhiskerRight_cons_whisker $e_η₁ $e_η₂ $e_ηs₁ $e_ηs₂ $e_η₃ $e_η₄ $e_η₅)
  mkEvalWhiskerRightComp g h η η₁ η₂ η₃ η₄ e_η₁ e_η₂ e_η₃ e_η₄ := do
    let ctx ← read
    let .some _monoidal := ctx.instMonoidal? | synthMonoidalError
    let f ← η.srcM
    let f' ← η.tgtM
    have f : Q($ctx.C) := f.e
    have f' : Q($ctx.C) := f'.e
    have g : Q($ctx.C) := g.e
    have h : Q($ctx.C) := h.e
    have η : Q($f ⟶ $f') := η.e.e
    have η₁ : Q($f ⊗ $g ⟶ $f' ⊗ $g) := η₁.e.e
    have η₂ : Q(($f ⊗ $g) ⊗ $h ⟶ ($f' ⊗ $g) ⊗ $h) := η₂.e.e
    have η₃ : Q(($f ⊗ $g) ⊗ $h ⟶ $f' ⊗ ($g ⊗ $h)) := η₃.e.e
    have η₄ : Q($f ⊗ ($g ⊗ $h) ⟶ $f' ⊗ ($g ⊗ $h)) := η₄.e.e
    have e_η₁ : Q($η ▷ $g = $η₁) := e_η₁
    have e_η₂ : Q($η₁ ▷ $h = $η₂) := e_η₂
    have e_η₃ : Q($η₂ ≫ (α_ _ _ _).hom = $η₃) := e_η₃
    have e_η₄ : Q((α_ _ _ _).inv ≫ $η₃ = $η₄) := e_η₄
    return q(evalWhiskerRight_comp $e_η₁ $e_η₂ $e_η₃ $e_η₄)
  mkEvalWhiskerRightId η η₁ η₂ e_η₁ e_η₂ := do
    let ctx ← read
    let .some _monoidal := ctx.instMonoidal? | synthMonoidalError
    let f ← η.srcM
    let g ← η.tgtM
    have f : Q($ctx.C) := f.e
    have g : Q($ctx.C) := g.e
    have η : Q($f ⟶ $g) := η.e.e
    have η₁ : Q($f ⟶ $g ⊗ 𝟙_ _) := η₁.e.e
    have η₂ : Q($f ⊗ 𝟙_ _ ⟶ $g ⊗ 𝟙_ _) := η₂.e.e
    have e_η₁ : Q($η ≫ (ρ_ _).inv = $η₁) := e_η₁
    have e_η₂ : Q((ρ_ _).hom ≫ $η₁ = $η₂) := e_η₂
    return q(evalWhiskerRight_id $e_η₁ $e_η₂)


instance : MkEvalHorizontalComp MonoidalM where
  mkEvalHorizontalCompAuxOf η θ := do
    let ctx ← read
    let .some _monoidal := ctx.instMonoidal? | synthMonoidalError
    let f ← η.srcM
    let g ← η.tgtM
    let h ← θ.srcM
    let i ← θ.tgtM
    have f : Q($ctx.C) := f.e
    have g : Q($ctx.C) := g.e
    have h : Q($ctx.C) := h.e
    have i : Q($ctx.C) := i.e
    have η : Q($f ⟶ $g) := η.e.e
    have θ : Q($h ⟶ $i) := θ.e.e
    return q(evalHorizontalCompAux_of $η $θ)
  mkEvalHorizontalCompAuxCons η ηs θ ηθ η₁ ηθ₁ ηθ₂ e_ηθ e_η₁ e_ηθ₁ e_ηθ₂ := do
    let ctx ← read
    let .some _monoidal := ctx.instMonoidal? | synthMonoidalError
    let f ← η.srcM
    let g ← η.tgtM
    let f' ← ηs.srcM
    let g' ← ηs.tgtM
    let h ← θ.srcM
    let i ← θ.tgtM
    have f : Q($ctx.C) := f.e
    have g : Q($ctx.C) := g.e
    have f' : Q($ctx.C) := f'.e
    have g' : Q($ctx.C) := g'.e
    have h : Q($ctx.C) := h.e
    have i : Q($ctx.C) := i.e
    have η : Q($f ⟶ $g) := η.e.e
    have ηs : Q($f' ⟶ $g') := ηs.e.e
    have θ : Q($h ⟶ $i) := θ.e.e
    have ηθ : Q($f' ⊗ $h ⟶ $g' ⊗ $i) := ηθ.e.e
    have η₁ : Q($f ⊗ ($f' ⊗ $h) ⟶ $g ⊗ ($g' ⊗ $i)) := η₁.e.e
    have ηθ₁ : Q($f ⊗ ($f' ⊗ $h) ⟶ ($g ⊗ $g') ⊗ $i) := ηθ₁.e.e
    have ηθ₂ : Q(($f ⊗ $f') ⊗ $h ⟶ ($g ⊗ $g') ⊗ $i) := ηθ₂.e.e
    have e_ηθ : Q($ηs ⊗ $θ = $ηθ) := e_ηθ
    have e_η₁ : Q(((Iso.refl _).hom ≫ $η ≫ (Iso.refl _).hom) ⊗ $ηθ = $η₁) := e_η₁
    have e_ηθ₁ : Q($η₁ ≫ (α_ _ _ _).inv = $ηθ₁) := e_ηθ₁
    have e_ηθ₂ : Q((α_ _ _ _).hom ≫ $ηθ₁ = $ηθ₂) := e_ηθ₂
    return q(evalHorizontalCompAux_cons $e_ηθ $e_η₁ $e_ηθ₁ $e_ηθ₂)
  mkEvalHorizontalCompAux'Whisker f η θ ηθ η₁ η₂ η₃ e_ηθ e_η₁ e_η₂ e_η₃ := do
    let ctx ← read
    let .some _monoidal := ctx.instMonoidal? | synthMonoidalError
    let g ← η.srcM
    let h ← η.tgtM
    let f' ← θ.srcM
    let g' ← θ.tgtM
    have f : Q($ctx.C) := f.e
    have g : Q($ctx.C) := g.e
    have h : Q($ctx.C) := h.e
    have f' : Q($ctx.C) := f'.e
    have g' : Q($ctx.C) := g'.e
    have η : Q($g ⟶ $h) := η.e.e
    have θ : Q($f' ⟶ $g') := θ.e.e
    have ηθ : Q($g ⊗ $f' ⟶ $h ⊗ $g') := ηθ.e.e
    have η₁ : Q($f ⊗ ($g ⊗ $f') ⟶ $f ⊗ ($h ⊗ $g')) := η₁.e.e
    have η₂ : Q($f ⊗ ($g ⊗ $f') ⟶ ($f ⊗ $h) ⊗ $g') := η₂.e.e
    have η₃ : Q(($f ⊗ $g) ⊗ $f' ⟶ ($f ⊗ $h) ⊗ $g') := η₃.e.e
    have e_ηθ : Q($η ⊗ $θ = $ηθ) := e_ηθ
    have e_η₁ : Q($f ◁ $ηθ = $η₁) := e_η₁
    have e_η₂ : Q($η₁ ≫ (α_ _ _ _).inv = $η₂) := e_η₂
    have e_η₃ : Q((α_ _ _ _).hom ≫ $η₂ = $η₃) := e_η₃
    return q(evalHorizontalCompAux'_whisker $e_ηθ $e_η₁ $e_η₂ $e_η₃)
  mkEvalHorizontalCompAux'OfWhisker f η θ η₁ ηθ ηθ₁ ηθ₂ e_η₁ e_ηθ e_ηθ₁ e_ηθ₂ := do
    let ctx ← read
    let .some _monoidal := ctx.instMonoidal? | synthMonoidalError
    let g ← η.srcM
    let h ← η.tgtM
    let f' ← θ.srcM
    let g' ← θ.tgtM
    have f : Q($ctx.C) := f.e
    have g : Q($ctx.C) := g.e
    have h : Q($ctx.C) := h.e
    have f' : Q($ctx.C) := f'.e
    have g' : Q($ctx.C) := g'.e
    have η : Q($g ⟶ $h) := η.e.e
    have θ : Q($f' ⟶ $g') := θ.e.e
    have η₁ : Q($g ⊗ $f ⟶ $h ⊗ $f) := η₁.e.e
    have ηθ : Q(($g ⊗ $f) ⊗ $f' ⟶ ($h ⊗ $f) ⊗ $g') := ηθ.e.e
    have ηθ₁ : Q(($g ⊗ $f) ⊗ $f' ⟶ $h ⊗ ($f ⊗ $g')) := ηθ₁.e.e
    have ηθ₂ : Q($g ⊗ ($f ⊗ $f') ⟶ $h ⊗ ($f ⊗ $g')) := ηθ₂.e.e
    have e_η₁ : Q($η ▷ $f = $η₁) := e_η₁
    have e_ηθ : Q($η₁ ⊗ ((Iso.refl _).hom ≫ $θ ≫ (Iso.refl _).hom) = $ηθ) := e_ηθ
    have e_ηθ₁ : Q($ηθ ≫ (α_ _ _ _).hom = $ηθ₁) := e_ηθ₁
    have e_ηθ₂ : Q((α_ _ _ _).inv ≫ $ηθ₁ = $ηθ₂) := e_ηθ₂
    return q(evalHorizontalCompAux'_of_whisker $e_η₁ $e_ηθ $e_ηθ₁ $e_ηθ₂)
  mkEvalHorizontalCompNilNil α β := do
    let ctx ← read
    let .some _monoidal := ctx.instMonoidal? | synthMonoidalError
    let f ← α.srcM
    let g ← α.tgtM
    let h ← β.srcM
    let i ← β.tgtM
    have f : Q($ctx.C) := f.e
    have g : Q($ctx.C) := g.e
    have h : Q($ctx.C) := h.e
    have i : Q($ctx.C) := i.e
    have α : Q($f ≅ $g) := α.e
    have β : Q($h ≅ $i) := β.e
    return q(evalHorizontalComp_nil_nil $α $β)
  mkEvalHorizontalCompNilCons α β η ηs η₁ ηs₁ η₂ η₃ e_η₁ e_ηs₁ e_η₂ e_η₃ := do
    let ctx ← read
    let .some _monoidal := ctx.instMonoidal? | synthMonoidalError
    let f ← α.srcM
    let g ← α.tgtM
    let f' ← β.srcM
    let g' ← β.tgtM
    let h ← η.tgtM
    let i ← ηs.tgtM
    have f : Q($ctx.C) := f.e
    have g : Q($ctx.C) := g.e
    have f' : Q($ctx.C) := f'.e
    have g' : Q($ctx.C) := g'.e
    have h : Q($ctx.C) := h.e
    have i : Q($ctx.C) := i.e
    have α : Q($f ≅ $g) := α.e
    have β : Q($f' ≅ $g') := β.e
    have η : Q($g' ⟶ $h) := η.e.e
    have ηs : Q($h ⟶ $i) := ηs.e.e
    have η₁ : Q($g ⊗ $g' ⟶ $g ⊗ $h) := η₁.e.e
    have ηs₁ : Q($g ⊗ $h ⟶ $g ⊗ $i) := ηs₁.e.e
    have η₂ : Q($g ⊗ $g' ⟶ $g ⊗ $i) := η₂.e.e
    have η₃ : Q($f ⊗ $f' ⟶ $g ⊗ $i) := η₃.e.e
    have e_η₁ : Q($g ◁ ((Iso.refl _).hom ≫ $η ≫ (Iso.refl _).hom) = $η₁) := e_η₁
    have e_ηs₁ : Q($g ◁ $ηs = $ηs₁) := e_ηs₁
    have e_η₂ : Q($η₁ ≫ $ηs₁ = $η₂) := e_η₂
    have e_η₃ : Q(($α ⊗ $β).hom ≫ $η₂ = $η₃) := e_η₃
    return q(evalHorizontalComp_nil_cons $e_η₁ $e_ηs₁ $e_η₂ $e_η₃)
  mkEvalHorizontalCompConsNil α β η ηs η₁ ηs₁ η₂ η₃ e_η₁ e_ηs₁ e_η₂ e_η₃ := do
    let ctx ← read
    let .some _monoidal := ctx.instMonoidal? | synthMonoidalError
    let f ← α.srcM
    let g ← α.tgtM
    let h ← η.tgtM
    let i ← ηs.tgtM
    let f' ← β.srcM
    let g' ← β.tgtM
    have f : Q($ctx.C) := f.e
    have g : Q($ctx.C) := g.e
    have h : Q($ctx.C) := h.e
    have i : Q($ctx.C) := i.e
    have f' : Q($ctx.C) := f'.e
    have g' : Q($ctx.C) := g'.e
    have α : Q($f ≅ $g) := α.e
    have η : Q($g ⟶ $h) := η.e.e
    have ηs : Q($h ⟶ $i) := ηs.e.e
    have β : Q($f' ≅ $g') := β.e
    have η₁ : Q($g ⊗ $g' ⟶ $h ⊗ $g') := η₁.e.e
    have ηs₁ : Q($h ⊗ $g' ⟶ $i ⊗ $g') := ηs₁.e.e
    have η₂ : Q($g ⊗ $g' ⟶ $i ⊗ $g') := η₂.e.e
    have η₃ : Q($f ⊗ $f' ⟶ $i ⊗ $g') := η₃.e.e
    have e_η₁ : Q(((Iso.refl _).hom ≫ $η ≫ (Iso.refl _).hom) ▷ $g' = $η₁) := e_η₁
    have e_ηs₁ : Q($ηs ▷ $g' = $ηs₁) := e_ηs₁
    have e_η₂ : Q($η₁ ≫ $ηs₁ = $η₂) := e_η₂
    have e_η₃ : Q(($α ⊗ $β).hom ≫ $η₂ = $η₃) := e_η₃
    return q(evalHorizontalComp_cons_nil $e_η₁ $e_ηs₁ $e_η₂ $e_η₃)
  mkEvalHorizontalCompConsCons α β η θ ηs θs ηθ ηθs ηθ₁ ηθ₂ e_ηθ e_ηθs e_ηθ₁ e_ηθ₂ := do
    let ctx ← read
    let .some _monoidal := ctx.instMonoidal? | synthMonoidalError
    let f ← α.srcM
    let g ← α.tgtM
    let h ← η.tgtM
    let i ← ηs.tgtM
    let f' ← β.srcM
    let g' ← β.tgtM
    let h' ← θ.tgtM
    let i' ← θs.tgtM
    have f : Q($ctx.C) := f.e
    have g : Q($ctx.C) := g.e
    have h : Q($ctx.C) := h.e
    have i : Q($ctx.C) := i.e
    have f' : Q($ctx.C) := f'.e
    have g' : Q($ctx.C) := g'.e
    have h' : Q($ctx.C) := h'.e
    have i' : Q($ctx.C) := i'.e
    have α : Q($f ≅ $g) := α.e
    have η : Q($g ⟶ $h) := η.e.e
    have ηs : Q($h ⟶ $i) := ηs.e.e
    have β : Q($f' ≅ $g') := β.e
    have θ : Q($g' ⟶ $h') := θ.e.e
    have θs : Q($h' ⟶ $i') := θs.e.e
    have ηθ : Q($g ⊗ $g' ⟶ $h ⊗ $h') := ηθ.e.e
    have ηθs : Q($h ⊗ $h' ⟶ $i ⊗ $i') := ηθs.e.e
    have ηθ₁ : Q($g ⊗ $g' ⟶ $i ⊗ $i') := ηθ₁.e.e
    have ηθ₂ : Q($f ⊗ $f' ⟶ $i ⊗ $i') := ηθ₂.e.e
    have e_ηθ : Q($η ⊗ $θ = $ηθ) := e_ηθ
    have e_ηθs : Q($ηs ⊗ $θs = $ηθs) := e_ηθs
    have e_ηθ₁ : Q($ηθ ≫ $ηθs = $ηθ₁) := e_ηθ₁
    have e_ηθ₂ : Q(($α ⊗ $β).hom ≫ $ηθ₁ = $ηθ₂) := e_ηθ₂
    return q(evalHorizontalComp_cons_cons $e_ηθ $e_ηθs $e_ηθ₁ $e_ηθ₂)


instance : MkEval MonoidalM where
  mkEvalComp η θ η' θ' ι e_η e_θ e_ηθ := do
    let ctx ← read
    let _cat := ctx.instCat
    let f ← η'.srcM
    let g ← η'.tgtM
    let h ← θ'.tgtM
    have f : Q($ctx.C) := f.e
    have g : Q($ctx.C) := g.e
    have h : Q($ctx.C) := h.e
    have η : Q($f ⟶ $g) := η.e
    have η' : Q($f ⟶ $g) := η'.e.e
    have θ : Q($g ⟶ $h) := θ.e
    have θ' : Q($g ⟶ $h) := θ'.e.e
    have ι : Q($f ⟶ $h) := ι.e.e
    have e_η : Q($η = $η') := e_η
    have e_θ : Q($θ = $θ') := e_θ
    have e_ηθ : Q($η' ≫ $θ' = $ι) := e_ηθ
    return q(eval_comp $e_η $e_θ $e_ηθ)
  mkEvalWhiskerLeft f η η' θ e_η e_θ := do
    let ctx ← read
    let .some _monoidal := ctx.instMonoidal? | synthMonoidalError
    let g ← η'.srcM
    let h ← η'.tgtM
    have f : Q($ctx.C) := f.e
    have g : Q($ctx.C) := g.e
    have h : Q($ctx.C) := h.e
    have η : Q($g ⟶ $h) := η.e
    have η' : Q($g ⟶ $h) := η'.e.e
    have θ : Q($f ⊗ $g ⟶ $f ⊗ $h) := θ.e.e
    have e_η : Q($η = $η') := e_η
    have e_θ : Q($f ◁ $η' = $θ) := e_θ
    return q(eval_whiskerLeft $e_η $e_θ)
  mkEvalWhiskerRight η h η' θ e_η e_θ := do
    let ctx ← read
    let .some _monoidal := ctx.instMonoidal? | synthMonoidalError
    let f ← η'.srcM
    let g ← η'.tgtM
    have f : Q($ctx.C) := f.e
    have g : Q($ctx.C) := g.e
    have h : Q($ctx.C) := h.e
    have η : Q($f ⟶ $g) := η.e
    have η' : Q($f ⟶ $g) := η'.e.e
    have θ : Q($f ⊗ $h ⟶ $g ⊗ $h) := θ.e.e
    have e_η : Q($η = $η') := e_η
    have e_θ : Q($η' ▷ $h = $θ) := e_θ
    return q(eval_whiskerRight $e_η $e_θ)
  mkEvalHorizontalComp η θ η' θ' ι e_η e_θ e_ι := do
    let ctx ← read
    let .some _monoidal := ctx.instMonoidal? | synthMonoidalError
    let f ← η'.srcM
    let g ← η'.tgtM
    let h ← θ'.srcM
    let i ← θ'.tgtM
    have f : Q($ctx.C) := f.e
    have g : Q($ctx.C) := g.e
    have h : Q($ctx.C) := h.e
    have i : Q($ctx.C) := i.e
    have η : Q($f ⟶ $g) := η.e
    have η' : Q($f ⟶ $g) := η'.e.e
    have θ : Q($h ⟶ $i) := θ.e
    have θ' : Q($h ⟶ $i) := θ'.e.e
    have ι : Q($f ⊗ $h ⟶ $g ⊗ $i) := ι.e.e
    have e_η : Q($η = $η') := e_η
    have e_θ : Q($θ = $θ') := e_θ
    have e_ι : Q($η' ⊗ $θ' = $ι) := e_ι
    return q(eval_tensorHom $e_η $e_θ $e_ι)
  mkEvalOf η := do
    let ctx ← read
    let _cat := ctx.instCat
    let f := η.src
    let g := η.tgt
    have f : Q($ctx.C) := f.e
    have g : Q($ctx.C) := g.e
    have η : Q($f ⟶ $g) := η.e
    return q(eval_of $η)
  mkEvalMonoidalComp η θ α η' θ' αθ ηαθ e_η e_θ e_αθ e_ηαθ := do
    let ctx ← read
    let _cat := ctx.instCat
    let f ← η'.srcM
    let g ← η'.tgtM
    let h ← α.tgtM
    let i ← θ'.tgtM
    have f : Q($ctx.C) := f.e
    have g : Q($ctx.C) := g.e
    have h : Q($ctx.C) := h.e
    have i : Q($ctx.C) := i.e
    have η : Q($f ⟶ $g) := η.e
    have η' : Q($f ⟶ $g) := η'.e.e
    have α : Q($g ≅ $h) := α.e
    have θ : Q($h ⟶ $i) := θ.e
    have θ' : Q($h ⟶ $i) := θ'.e.e
    have αθ : Q($g ⟶ $i) := αθ.e.e
    have ηαθ : Q($f ⟶ $i) := ηαθ.e.e
    have e_η : Q($η = $η') := e_η
    have e_θ : Q($θ = $θ') := e_θ
    have e_αθ : Q(Iso.hom $α ≫ $θ' = $αθ) := e_αθ
    have e_ηαθ : Q($η' ≫ $αθ = $ηαθ) := e_ηαθ
    return q(eval_monoidalComp $e_η $e_θ $e_αθ $e_ηαθ)


instance : MonadNormalExpr MonoidalM where
  whiskerRightM η h := do
    return .whisker (← MonadMor₂.whiskerRightM η.e (.of h)) η h
  hConsM η θ := do
    return .cons (← MonadMor₂.horizontalCompM η.e θ.e) η θ
  whiskerLeftM f η := do
    return .whisker (← MonadMor₂.whiskerLeftM (.of f) η.e) f η
  nilM α := do
    return .nil (← MonadMor₂.homM α) α
  consM α η ηs := do
    return .cons (← MonadMor₂.comp₂M (← MonadMor₂.homM α) (← MonadMor₂.comp₂M η.e ηs.e)) α η ηs


instance : MkMor₂ MonoidalM where
  ofExpr := Mor₂OfExpr


