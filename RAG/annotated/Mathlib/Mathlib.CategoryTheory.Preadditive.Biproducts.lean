/-- In a preadditive category, we can construct a biproduct for `f : J → C` from
any bicone `b` for `f` satisfying `total : ∑ j : J, b.π j ≫ b.ι j = 𝟙 b.X`.

(That is, such a bicone is a limit cone and a colimit cocone.)
-/
def isBilimitOfTotal {f : J → C} (b : Bicone f) (total : ∑ j : J, b.π j ≫ b.ι j = 𝟙 b.pt) :
    b.IsBilimit where
  isLimit :=
    { lift := fun s => ∑ j : J, s.π.app ⟨j⟩ ≫ b.ι j
      uniq := fun s m h => by
        /-
          C : Type u
          inst✝² : CategoryTheory.Category.{v, u} C
          inst✝¹ : CategoryTheory.Preadditive C
          J : Type
          inst✝ : Fintype J
          f : J → C
          b : CategoryTheory.Limits.Bicone f
          total : Eq (Finset.univ.sum fun j => CategoryTheory.CategoryStruct.comp (b.π j …
          s : CategoryTheory.Limits.Cone (CategoryTheory.Discrete.functor f)
          m : Quiver.Hom s.pt b.toCone.pt
          h : ∀ (j : CategoryTheory.Discrete J), Eq (CategoryTheory.CategoryStruct.comp  …
          ⊢ Eq m ((fun s => Finset.univ.sum fun j => CategoryTheory.CategoryStruct.comp  …
        -/
        erw [← Category.comp_id m, ← total, comp_sum]
        /-
          C : Type u
          inst✝² : CategoryTheory.Category.{v, u} C
          inst✝¹ : CategoryTheory.Preadditive C
          J : Type
          inst✝ : Fintype J
          f : J → C
          b : CategoryTheory.Limits.Bicone f
          total : Eq (Finset.univ.sum fun j => CategoryTheory.CategoryStruct.comp (b.π j …
          s : CategoryTheory.Limits.Cone (CategoryTheory.Discrete.functor f)
          m : Quiver.Hom s.pt b.toCone.pt
          h : ∀ (j : CategoryTheory.Discrete J), Eq (CategoryTheory.CategoryStruct.comp  …
          ⊢ Eq (Finset.univ.sum fun j => CategoryTheory.CategoryStruct.comp m (CategoryT …
        -/
        apply Finset.sum_congr rfl
        /-
          C : Type u
          inst✝² : CategoryTheory.Category.{v, u} C
          inst✝¹ : CategoryTheory.Preadditive C
          J : Type
          inst✝ : Fintype J
          f : J → C
          b : CategoryTheory.Limits.Bicone f
          total : Eq (Finset.univ.sum fun j => CategoryTheory.CategoryStruct.comp (b.π j …
          s : CategoryTheory.Limits.Cone (CategoryTheory.Discrete.functor f)
          m : Quiver.Hom s.pt b.toCone.pt
          h : ∀ (j : CategoryTheory.Discrete J), Eq (CategoryTheory.CategoryStruct.comp  …
          ⊢ ∀ (x : J), Membership.mem Finset.univ x → Eq (CategoryTheory.CategoryStruct. …
        -/
        intro j _
        have reassoced : m ≫ Bicone.π b j ≫ Bicone.ι b j = s.π.app ⟨j⟩ ≫ Bicone.ι b j := by
          erw [← Category.assoc, eq_whisker (h ⟨j⟩)]
        /-
          C : Type u
          inst✝² : CategoryTheory.Category.{v, u} C
          inst✝¹ : CategoryTheory.Preadditive C
          J : Type
          inst✝ : Fintype J
          f : J → C
          b : CategoryTheory.Limits.Bicone f
          total : Eq (Finset.univ.sum fun j => CategoryTheory.CategoryStruct.comp (b.π j …
          s : CategoryTheory.Limits.Cone (CategoryTheory.Discrete.functor f)
          m : Quiver.Hom s.pt b.toCone.pt
          h : ∀ (j : CategoryTheory.Discrete J), Eq (CategoryTheory.CategoryStruct.comp  …
          j : J
          a✝ : Membership.mem Finset.univ j
          reassoced : Eq (CategoryTheory.CategoryStruct.comp m (CategoryTheory.CategoryS …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp m (CategoryTheory.CategoryStruct.comp …
        -/
        rw [reassoced]
        /-
          🎉 no goals
        -/
      fac := fun s j => by
        classical
        cases j
        simp only [sum_comp, Category.assoc, Bicone.toCone_π_app, b.ι_π, comp_dite]
        -- See note [dsimp, simp].
        dsimp
        simp }
  isColimit :=
    { desc := fun s => ∑ j : J, b.π j ≫ s.ι.app ⟨j⟩
      uniq := fun s m h => by
        /-
          C : Type u
          inst✝² : CategoryTheory.Category.{v, u} C
          inst✝¹ : CategoryTheory.Preadditive C
          J : Type
          inst✝ : Fintype J
          f : J → C
          b : CategoryTheory.Limits.Bicone f
          total : Eq (Finset.univ.sum fun j => CategoryTheory.CategoryStruct.comp (b.π j …
          s : CategoryTheory.Limits.Cocone (CategoryTheory.Discrete.functor f)
          m : Quiver.Hom b.toCocone.pt s.pt
          h : ∀ (j : CategoryTheory.Discrete J), Eq (CategoryTheory.CategoryStruct.comp  …
          ⊢ Eq m ((fun s => Finset.univ.sum fun j => CategoryTheory.CategoryStruct.comp  …
        -/
        erw [← Category.id_comp m, ← total, sum_comp]
        /-
          C : Type u
          inst✝² : CategoryTheory.Category.{v, u} C
          inst✝¹ : CategoryTheory.Preadditive C
          J : Type
          inst✝ : Fintype J
          f : J → C
          b : CategoryTheory.Limits.Bicone f
          total : Eq (Finset.univ.sum fun j => CategoryTheory.CategoryStruct.comp (b.π j …
          s : CategoryTheory.Limits.Cocone (CategoryTheory.Discrete.functor f)
          m : Quiver.Hom b.toCocone.pt s.pt
          h : ∀ (j : CategoryTheory.Discrete J), Eq (CategoryTheory.CategoryStruct.comp  …
          ⊢ Eq (Finset.univ.sum fun j => CategoryTheory.CategoryStruct.comp (CategoryThe …
        -/
        apply Finset.sum_congr rfl
        /-
          C : Type u
          inst✝² : CategoryTheory.Category.{v, u} C
          inst✝¹ : CategoryTheory.Preadditive C
          J : Type
          inst✝ : Fintype J
          f : J → C
          b : CategoryTheory.Limits.Bicone f
          total : Eq (Finset.univ.sum fun j => CategoryTheory.CategoryStruct.comp (b.π j …
          s : CategoryTheory.Limits.Cocone (CategoryTheory.Discrete.functor f)
          m : Quiver.Hom b.toCocone.pt s.pt
          h : ∀ (j : CategoryTheory.Discrete J), Eq (CategoryTheory.CategoryStruct.comp  …
          ⊢ ∀ (x : J), Membership.mem Finset.univ x → Eq (CategoryTheory.CategoryStruct. …
        -/
        intro j _
        /-
          C : Type u
          inst✝² : CategoryTheory.Category.{v, u} C
          inst✝¹ : CategoryTheory.Preadditive C
          J : Type
          inst✝ : Fintype J
          f : J → C
          b : CategoryTheory.Limits.Bicone f
          total : Eq (Finset.univ.sum fun j => CategoryTheory.CategoryStruct.comp (b.π j …
          s : CategoryTheory.Limits.Cocone (CategoryTheory.Discrete.functor f)
          m : Quiver.Hom b.toCocone.pt s.pt
          h : ∀ (j : CategoryTheory.Discrete J), Eq (CategoryTheory.CategoryStruct.comp  …
          j : J
          a✝ : Membership.mem Finset.univ j
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
        -/
        erw [Category.assoc, h ⟨j⟩]
        /-
          🎉 no goals
        -/
      fac := fun s j => by
        classical
        cases j
        simp only [comp_sum, ← Category.assoc, Bicone.toCocone_ι_app, b.ι_π, dite_comp]
        dsimp; simp }


theorem IsBilimit.total {f : J → C} {b : Bicone f} (i : b.IsBilimit) :
    ∑ j : J, b.π j ≫ b.ι j = 𝟙 b.pt :=
  i.isLimit.hom_ext fun j => by
    classical
    cases j
    simp [sum_comp, b.ι_π, comp_dite]


/-- In a preadditive category, we can construct a biproduct for `f : J → C` from
any bicone `b` for `f` satisfying `total : ∑ j : J, b.π j ≫ b.ι j = 𝟙 b.X`.

(That is, such a bicone is a limit cone and a colimit cocone.)
-/
theorem hasBiproduct_of_total {f : J → C} (b : Bicone f)
    (total : ∑ j : J, b.π j ≫ b.ι j = 𝟙 b.pt) : HasBiproduct f :=
  HasBiproduct.mk
    { bicone := b
      isBilimit := isBilimitOfTotal b total }


/-- In a preadditive category, any finite bicone which is a limit cone is in fact a bilimit
    bicone. -/
def isBilimitOfIsLimit {f : J → C} (t : Bicone f) (ht : IsLimit t.toCone) : t.IsBilimit :=
  isBilimitOfTotal _ <|
    ht.hom_ext fun j => by
      classical
      cases j
      simp [sum_comp, t.ι_π, dite_comp, comp_dite]


/-- We can turn any limit cone over a pair into a bilimit bicone. -/
def biconeIsBilimitOfLimitConeOfIsLimit {f : J → C} {t : Cone (Discrete.functor f)}
    (ht : IsLimit t) : (Bicone.ofLimitCone ht).IsBilimit :=
                                                                              /-
                                                                                C : Type u
                                                                                inst✝² : CategoryTheory.Category.{v, u} C
                                                                                inst✝¹ : CategoryTheory.Preadditive C
                                                                                J : Type
                                                                                inst✝ : Fintype J
                                                                                f : J → C
                                                                                t : CategoryTheory.Limits.Cone (CategoryTheory.Discrete.functor f)
                                                                                ht : CategoryTheory.Limits.IsLimit t
                                                                                ⊢ ∀ (j : CategoryTheory.Discrete J), Eq (t.π.app j) (CategoryTheory.CategorySt …
                                                                              -/
  isBilimitOfIsLimit _ <| IsLimit.ofIsoLimit ht <| Cones.ext (Iso.refl _) (by aesop_cat)
                                                                              /-
                                                                                🎉 no goals
                                                                              -/


/-- In a preadditive category, any finite bicone which is a colimit cocone is in fact a bilimit
    bicone. -/
def isBilimitOfIsColimit {f : J → C} (t : Bicone f) (ht : IsColimit t.toCocone) : t.IsBilimit :=
  isBilimitOfTotal _ <|
    ht.hom_ext fun j => by
      classical
      cases j
      simp_rw [Bicone.toCocone_ι_app, comp_sum, ← Category.assoc, t.ι_π, dite_comp]
      simp


/-- We can turn any limit cone over a pair into a bilimit bicone. -/
def biconeIsBilimitOfColimitCoconeOfIsColimit {f : J → C} {t : Cocone (Discrete.functor f)}
    (ht : IsColimit t) : (Bicone.ofColimitCocone ht).IsBilimit :=
  isBilimitOfIsColimit _ <| IsColimit.ofIsoColimit ht <| Cocones.ext (Iso.refl _) <| by
    /-
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      inst✝¹ : CategoryTheory.Preadditive C
      J : Type
      inst✝ : Fintype J
      f : J → C
      t : CategoryTheory.Limits.Cocone (CategoryTheory.Discrete.functor f)
      ht : CategoryTheory.Limits.IsColimit t
      ⊢ ∀ (j : CategoryTheory.Discrete J), Eq (CategoryTheory.CategoryStruct.comp (t …
    -/
    rintro ⟨j⟩; simp
                /-
                  🎉 no goals
                -/


/-- In a preadditive category, if the product over `f : J → C` exists,
    then the biproduct over `f` exists. -/
theorem HasBiproduct.of_hasProduct (f : J → C) [HasProduct f] : HasBiproduct f := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Preadditive C
    J : Type
    inst✝¹ : Finite J
    f : J → C
    inst✝ : CategoryTheory.Limits.HasProduct f
    ⊢ CategoryTheory.Limits.HasBiproduct f
  -/
  cases nonempty_fintype J
  exact HasBiproduct.mk
    { bicone := _
      isBilimit := biconeIsBilimitOfLimitConeOfIsLimit (limit.isLimit _) }


/-- In a preadditive category, if the coproduct over `f : J → C` exists,
    then the biproduct over `f` exists. -/
theorem HasBiproduct.of_hasCoproduct (f : J → C) [HasCoproduct f] : HasBiproduct f := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Preadditive C
    J : Type
    inst✝¹ : Finite J
    f : J → C
    inst✝ : CategoryTheory.Limits.HasCoproduct f
    ⊢ CategoryTheory.Limits.HasBiproduct f
  -/
  cases nonempty_fintype J
  exact HasBiproduct.mk
    { bicone := _
      isBilimit := biconeIsBilimitOfColimitCoconeOfIsColimit (colimit.isColimit _) }


/-- A preadditive category with finite products has finite biproducts. -/
theorem HasFiniteBiproducts.of_hasFiniteProducts [HasFiniteProducts C] : HasFiniteBiproducts C :=
  ⟨fun _ => { has_biproduct := fun _ => HasBiproduct.of_hasProduct _ }⟩


/-- A preadditive category with finite coproducts has finite biproducts. -/
theorem HasFiniteBiproducts.of_hasFiniteCoproducts [HasFiniteCoproducts C] :
    HasFiniteBiproducts C :=
  ⟨fun _ => { has_biproduct := fun _ => HasBiproduct.of_hasCoproduct _ }⟩


/-- In any preadditive category, any biproduct satisfies
`∑ j : J, biproduct.π f j ≫ biproduct.ι f j = 𝟙 (⨁ f)`
-/
@[simp]
theorem biproduct.total : ∑ j : J, biproduct.π f j ≫ biproduct.ι f j = 𝟙 (⨁ f) :=
  IsBilimit.total (biproduct.isBilimit _)


theorem biproduct.lift_eq {T : C} {g : ∀ j, T ⟶ f j} :
    biproduct.lift g = ∑ j, g j ≫ biproduct.ι f j := by
  classical
  ext j
  simp only [sum_comp, biproduct.ι_π, comp_dite, biproduct.lift_π, Category.assoc, comp_zero,
    Finset.sum_dite_eq', Finset.mem_univ, eqToHom_refl, Category.comp_id, if_true]


theorem biproduct.desc_eq {T : C} {g : ∀ j, f j ⟶ T} :
    biproduct.desc g = ∑ j, biproduct.π f j ≫ g j := by
  classical
  ext j
  simp [comp_sum, biproduct.ι_π_assoc, dite_comp]


@[reassoc]
theorem biproduct.lift_desc {T U : C} {g : ∀ j, T ⟶ f j} {h : ∀ j, f j ⟶ U} :
    biproduct.lift g ≫ biproduct.desc h = ∑ j : J, g j ≫ h j := by
  classical
  simp [biproduct.lift_eq, biproduct.desc_eq, comp_sum, sum_comp, biproduct.ι_π_assoc, comp_dite,
    dite_comp]


theorem biproduct.map_eq [HasFiniteBiproducts C] {f g : J → C} {h : ∀ j, f j ⟶ g j} :
    biproduct.map h = ∑ j : J, biproduct.π f j ≫ h j ≫ biproduct.ι g j := by
  classical
  ext
  simp [biproduct.ι_π, biproduct.ι_π_assoc, comp_sum, sum_comp, comp_dite, dite_comp]


@[reassoc]
theorem biproduct.lift_matrix {K : Type} [Finite K] [HasFiniteBiproducts C] {f : J → C} {g : K → C}
    {P} (x : ∀ j, P ⟶ f j) (m : ∀ j k, f j ⟶ g k) :
    biproduct.lift x ≫ biproduct.matrix m = biproduct.lift fun k => ∑ j, x j ≫ m j k := by
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.Preadditive C
    J : Type
    inst✝² : Fintype J
    K : Type
    inst✝¹ : Finite K
    inst✝ : CategoryTheory.Limits.HasFiniteBiproducts C
    f : J → C
    g : K → C
    P : C
    x : (j : J) → Quiver.Hom P (f j)
    m : (j : J) → (k : K) → Quiver.Hom (f j) (g k)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.biproduct.lift …
  -/
  ext
  /-
    case w
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.Preadditive C
    J : Type
    inst✝² : Fintype J
    K : Type
    inst✝¹ : Finite K
    inst✝ : CategoryTheory.Limits.HasFiniteBiproducts C
    f : J → C
    g : K → C
    P : C
    x : (j : J) → Quiver.Hom P (f j)
    m : (j : J) → (k : K) → Quiver.Hom (f j) (g k)
    j✝ : K
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  simp [biproduct.lift_desc]
  /-
    🎉 no goals
  -/


@[reassoc]
theorem biproduct.matrix_desc [Fintype K] {f : J → C} {g : K → C}
    (m : ∀ j k, f j ⟶ g k) {P} (x : ∀ k, g k ⟶ P) :
    biproduct.matrix m ≫ biproduct.desc x = biproduct.desc fun j => ∑ k, m j k ≫ x k := by
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.Preadditive C
    J K : Type
    inst✝² : Finite J
    inst✝¹ : CategoryTheory.Limits.HasFiniteBiproducts C
    inst✝ : Fintype K
    f : J → C
    g : K → C
    m : (j : J) → (k : K) → Quiver.Hom (f j) (g k)
    P : C
    x : (k : K) → Quiver.Hom (g k) P
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.biproduct.matr …
  -/
  ext
  /-
    case w
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.Preadditive C
    J K : Type
    inst✝² : Finite J
    inst✝¹ : CategoryTheory.Limits.HasFiniteBiproducts C
    inst✝ : Fintype K
    f : J → C
    g : K → C
    m : (j : J) → (k : K) → Quiver.Hom (f j) (g k)
    P : C
    x : (k : K) → Quiver.Hom (g k) P
    j✝ : J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.biproduct.ι f  …
  -/
  simp [lift_desc]
  /-
    🎉 no goals
  -/


@[reassoc]
theorem biproduct.matrix_map {f : J → C} {g : K → C} {h : K → C} (m : ∀ j k, f j ⟶ g k)
    (n : ∀ k, g k ⟶ h k) :
    biproduct.matrix m ≫ biproduct.map n = biproduct.matrix fun j k => m j k ≫ n k := by
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.Preadditive C
    J K : Type
    inst✝² : Finite J
    inst✝¹ : CategoryTheory.Limits.HasFiniteBiproducts C
    inst✝ : Finite K
    f : J → C
    g h : K → C
    m : (j : J) → (k : K) → Quiver.Hom (f j) (g k)
    n : (k : K) → Quiver.Hom (g k) (h k)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.biproduct.matr …
  -/
  ext
  /-
    case w.w
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.Preadditive C
    J K : Type
    inst✝² : Finite J
    inst✝¹ : CategoryTheory.Limits.HasFiniteBiproducts C
    inst✝ : Finite K
    f : J → C
    g h : K → C
    m : (j : J) → (k : K) → Quiver.Hom (f j) (g k)
    n : (k : K) → Quiver.Hom (g k) (h k)
    j✝¹ : K
    j✝ : J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.biproduct.ι f  …
  -/
  simp
  /-
    🎉 no goals
  -/


@[reassoc]
theorem biproduct.map_matrix {f : J → C} {g : J → C} {h : K → C} (m : ∀ k, f k ⟶ g k)
    (n : ∀ j k, g j ⟶ h k) :
    biproduct.map m ≫ biproduct.matrix n = biproduct.matrix fun j k => m j ≫ n j k := by
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.Preadditive C
    J K : Type
    inst✝² : Finite J
    inst✝¹ : CategoryTheory.Limits.HasFiniteBiproducts C
    inst✝ : Finite K
    f g : J → C
    h : K → C
    m : (k : J) → Quiver.Hom (f k) (g k)
    n : (j : J) → (k : K) → Quiver.Hom (g j) (h k)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.biproduct.map  …
  -/
  ext
  /-
    case w.w
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.Preadditive C
    J K : Type
    inst✝² : Finite J
    inst✝¹ : CategoryTheory.Limits.HasFiniteBiproducts C
    inst✝ : Finite K
    f g : J → C
    h : K → C
    m : (k : J) → Quiver.Hom (f k) (g k)
    n : (j : J) → (k : K) → Quiver.Hom (g j) (h k)
    j✝¹ : K
    j✝ : J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.biproduct.ι f  …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- Reindex a categorical biproduct via an equivalence of the index types. -/
@[simps]
def biproduct.reindex {β γ : Type} [Finite β] (ε : β ≃ γ)
    (f : γ → C) [HasBiproduct f] [HasBiproduct (f ∘ ε)] : ⨁ f ∘ ε ≅ ⨁ f where
  hom := biproduct.desc fun b => biproduct.ι f (ε b)
  inv := biproduct.lift fun b => biproduct.π f (ε b)
  hom_inv_id := by
    /-
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      inst✝³ : CategoryTheory.Preadditive C
      β γ : Type
      inst✝² : Finite β
      ε : Equiv β γ
      f : γ → C
      inst✝¹ : CategoryTheory.Limits.HasBiproduct f
      inst✝ : CategoryTheory.Limits.HasBiproduct (Function.comp f ⇑ε)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.biproduct.desc …
    -/
    ext b b'
    /-
      case w.w
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      inst✝³ : CategoryTheory.Preadditive C
      β γ : Type
      inst✝² : Finite β
      ε : Equiv β γ
      f : γ → C
      inst✝¹ : CategoryTheory.Limits.HasBiproduct f
      inst✝ : CategoryTheory.Limits.HasBiproduct (Function.comp f ⇑ε)
      b b' : β
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.biproduct.ι (F …
    -/
    by_cases h : b' = b
      /-
        case pos
        C : Type u
        inst✝⁴ : CategoryTheory.Category.{v, u} C
        inst✝³ : CategoryTheory.Preadditive C
        β γ : Type
        inst✝² : Finite β
        ε : Equiv β γ
        f : γ → C
        inst✝¹ : CategoryTheory.Limits.HasBiproduct f
        inst✝ : CategoryTheory.Limits.HasBiproduct (Function.comp f ⇑ε)
        b b' : β
        h : Eq b' b
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.biproduct.ι (F …
      -/
    · subst h; simp
               /-
                 🎉 no goals
               -/
      /-
        case neg
        C : Type u
        inst✝⁴ : CategoryTheory.Category.{v, u} C
        inst✝³ : CategoryTheory.Preadditive C
        β γ : Type
        inst✝² : Finite β
        ε : Equiv β γ
        f : γ → C
        inst✝¹ : CategoryTheory.Limits.HasBiproduct f
        inst✝ : CategoryTheory.Limits.HasBiproduct (Function.comp f ⇑ε)
        b b' : β
        h : Not (Eq b' b)
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.biproduct.ι (F …
      -/
    · have : ε b' ≠ ε b := by simp [h]
      /-
        case neg
        C : Type u
        inst✝⁴ : CategoryTheory.Category.{v, u} C
        inst✝³ : CategoryTheory.Preadditive C
        β γ : Type
        inst✝² : Finite β
        ε : Equiv β γ
        f : γ → C
        inst✝¹ : CategoryTheory.Limits.HasBiproduct f
        inst✝ : CategoryTheory.Limits.HasBiproduct (Function.comp f ⇑ε)
        b b' : β
        h : Not (Eq b' b)
        this : Ne (ε b') (ε b)
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.biproduct.ι (F …
      -/
      simp [biproduct.ι_π_ne _ h, biproduct.ι_π_ne _ this]
      /-
        🎉 no goals
      -/
  inv_hom_id := by
    classical
    cases nonempty_fintype β
    ext g g'
    by_cases h : g' = g <;>
      simp [Preadditive.sum_comp, Preadditive.comp_sum, biproduct.lift_desc,
        biproduct.ι_π, biproduct.ι_π_assoc, comp_dite, Equiv.apply_eq_iff_eq_symm_apply,
        Finset.sum_dite_eq' Finset.univ (ε.symm g') _, h]


/-- In a preadditive category, we can construct a binary biproduct for `X Y : C` from
any binary bicone `b` satisfying `total : b.fst ≫ b.inl + b.snd ≫ b.inr = 𝟙 b.X`.

(That is, such a bicone is a limit cone and a colimit cocone.)
-/
def isBinaryBilimitOfTotal {X Y : C} (b : BinaryBicone X Y)
    (total : b.fst ≫ b.inl + b.snd ≫ b.inr = 𝟙 b.pt) : b.IsBilimit where
  isLimit :=
    { lift := fun s =>
      (BinaryFan.fst s ≫ b.inl : s.pt ⟶ b.pt) + (BinaryFan.snd s ≫ b.inr : s.pt ⟶ b.pt)
      uniq := fun s m h => by
        have reassoced (j : WalkingPair) {W : C} (h' : _ ⟶ W) :
          m ≫ b.toCone.π.app ⟨j⟩ ≫ h' = s.π.app ⟨j⟩ ≫ h' := by
            rw [← Category.assoc, eq_whisker (h ⟨j⟩)]
        erw [← Category.comp_id m, ← total, comp_add, reassoced WalkingPair.left,
          reassoced WalkingPair.right]
                           /-
                             C : Type u
                             inst✝¹ : CategoryTheory.Category.{v, u} C
                             inst✝ : CategoryTheory.Preadditive C
                             X Y : C
                             b : CategoryTheory.Limits.BinaryBicone X Y
                             total : Eq (HAdd.hAdd (CategoryTheory.CategoryStruct.comp b.fst b.inl) (Catego …
                             s : CategoryTheory.Limits.Cone (CategoryTheory.Limits.pair X Y)
                             j : CategoryTheory.Discrete CategoryTheory.Limits.WalkingPair
                             ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun s => HAdd.hAdd (CategoryTheory. …
                           -/
                                                  /-
                                                    🎉 no goals
                                                  -/
      fac := fun s j => by rcases j with ⟨⟨⟩⟩ <;> simp }
                                                  /-
                                                    🎉 no goals
                                                  -/
  isColimit :=
    { desc := fun s =>
        (b.fst ≫ BinaryCofan.inl s : b.pt ⟶ s.pt) + (b.snd ≫ BinaryCofan.inr s : b.pt ⟶ s.pt)
      uniq := fun s m h => by
        erw [← Category.id_comp m, ← total, add_comp, Category.assoc, Category.assoc,
          h ⟨WalkingPair.left⟩, h ⟨WalkingPair.right⟩]
                           /-
                             C : Type u
                             inst✝¹ : CategoryTheory.Category.{v, u} C
                             inst✝ : CategoryTheory.Preadditive C
                             X Y : C
                             b : CategoryTheory.Limits.BinaryBicone X Y
                             total : Eq (HAdd.hAdd (CategoryTheory.CategoryStruct.comp b.fst b.inl) (Catego …
                             s : CategoryTheory.Limits.Cocone (CategoryTheory.Limits.pair X Y)
                             j : CategoryTheory.Discrete CategoryTheory.Limits.WalkingPair
                             ⊢ Eq (CategoryTheory.CategoryStruct.comp (b.toCocone.ι.app j) ((fun s => HAdd. …
                           -/
                                                  /-
                                                    🎉 no goals
                                                  -/
      fac := fun s j => by rcases j with ⟨⟨⟩⟩ <;> simp }
                                                  /-
                                                    🎉 no goals
                                                  -/


theorem IsBilimit.binary_total {X Y : C} {b : BinaryBicone X Y} (i : b.IsBilimit) :
    b.fst ≫ b.inl + b.snd ≫ b.inr = 𝟙 b.pt :=
                                /-
                                  C : Type u
                                  inst✝¹ : CategoryTheory.Category.{v, u} C
                                  inst✝ : CategoryTheory.Preadditive C
                                  X Y : C
                                  b : CategoryTheory.Limits.BinaryBicone X Y
                                  i : b.IsBilimit
                                  j : CategoryTheory.Discrete CategoryTheory.Limits.WalkingPair
                                  ⊢ Eq (CategoryTheory.CategoryStruct.comp (HAdd.hAdd (CategoryTheory.CategorySt …
                                -/
                                                       /-
                                                         🎉 no goals
                                                       -/
  i.isLimit.hom_ext fun j => by rcases j with ⟨⟨⟩⟩ <;> simp
                                                       /-
                                                         🎉 no goals
                                                       -/


/-- In a preadditive category, we can construct a binary biproduct for `X Y : C` from
any binary bicone `b` satisfying `total : b.fst ≫ b.inl + b.snd ≫ b.inr = 𝟙 b.X`.

(That is, such a bicone is a limit cone and a colimit cocone.)
-/
theorem hasBinaryBiproduct_of_total {X Y : C} (b : BinaryBicone X Y)
    (total : b.fst ≫ b.inl + b.snd ≫ b.inr = 𝟙 b.pt) : HasBinaryBiproduct X Y :=
  HasBinaryBiproduct.mk
    { bicone := b
      isBilimit := isBinaryBilimitOfTotal b total }


/-- We can turn any limit cone over a pair into a bicone. -/
@[simps]
def BinaryBicone.ofLimitCone {X Y : C} {t : Cone (pair X Y)} (ht : IsLimit t) :
    BinaryBicone X Y where
  pt := t.pt
  fst := t.π.app ⟨WalkingPair.left⟩
  snd := t.π.app ⟨WalkingPair.right⟩
  inl := ht.lift (BinaryFan.mk (𝟙 X) 0)
  inr := ht.lift (BinaryFan.mk 0 (𝟙 Y))


theorem inl_of_isLimit {X Y : C} {t : BinaryBicone X Y} (ht : IsLimit t.toCone) :
    t.inl = ht.lift (BinaryFan.mk (𝟙 X) 0) := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    X Y : C
    t : CategoryTheory.Limits.BinaryBicone X Y
    ht : CategoryTheory.Limits.IsLimit t.toCone
    ⊢ Eq t.inl (ht.lift (CategoryTheory.Limits.BinaryFan.mk (CategoryTheory.Catego …
  -/
                                                                  /-
                                                                    🎉 no goals
                                                                  -/
  apply ht.uniq (BinaryFan.mk (𝟙 X) 0); rintro ⟨⟨⟩⟩ <;> dsimp <;> simp
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


theorem inr_of_isLimit {X Y : C} {t : BinaryBicone X Y} (ht : IsLimit t.toCone) :
    t.inr = ht.lift (BinaryFan.mk 0 (𝟙 Y)) := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    X Y : C
    t : CategoryTheory.Limits.BinaryBicone X Y
    ht : CategoryTheory.Limits.IsLimit t.toCone
    ⊢ Eq t.inr (ht.lift (CategoryTheory.Limits.BinaryFan.mk 0 (CategoryTheory.Cate …
  -/
                                                                  /-
                                                                    🎉 no goals
                                                                  -/
  apply ht.uniq (BinaryFan.mk 0 (𝟙 Y)); rintro ⟨⟨⟩⟩ <;> dsimp <;> simp
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


/-- In a preadditive category, any binary bicone which is a limit cone is in fact a bilimit
    bicone. -/
def isBinaryBilimitOfIsLimit {X Y : C} (t : BinaryBicone X Y) (ht : IsLimit t.toCone) :
    t.IsBilimit :=
                               /-
                                 C : Type u
                                 inst✝¹ : CategoryTheory.Category.{v, u} C
                                 inst✝ : CategoryTheory.Preadditive C
                                 X Y : C
                                 t : CategoryTheory.Limits.BinaryBicone X Y
                                 ht : CategoryTheory.Limits.IsLimit t.toCone
                                 ⊢ Eq (HAdd.hAdd (CategoryTheory.CategoryStruct.comp t.fst t.inl) (CategoryTheo …
                               -/
                                                                             /-
                                                                               🎉 no goals
                                                                             -/
  isBinaryBilimitOfTotal _ (by refine BinaryFan.IsLimit.hom_ext ht ?_ ?_ <;> simp)
                                                                             /-
                                                                               🎉 no goals
                                                                             -/


/-- We can turn any limit cone over a pair into a bilimit bicone. -/
def binaryBiconeIsBilimitOfLimitConeOfIsLimit {X Y : C} {t : Cone (pair X Y)} (ht : IsLimit t) :
    (BinaryBicone.ofLimitCone ht).IsBilimit :=
                                                               /-
                                                                 C : Type u
                                                                 inst✝¹ : CategoryTheory.Category.{v, u} C
                                                                 inst✝ : CategoryTheory.Preadditive C
                                                                 X Y : C
                                                                 t : CategoryTheory.Limits.Cone (CategoryTheory.Limits.pair X Y)
                                                                 ht : CategoryTheory.Limits.IsLimit t
                                                                 ⊢ Eq (CategoryTheory.CategoryStruct.comp (HAdd.hAdd (CategoryTheory.CategorySt …
                                                               -/
                                                               /-
                                                                 🎉 no goals
                                                               -/
  isBinaryBilimitOfTotal _ <| BinaryFan.IsLimit.hom_ext ht (by simp) (by simp)
                                                                         /-
                                                                           🎉 no goals
                                                                         -/


/-- In a preadditive category, if the product of `X` and `Y` exists, then the
    binary biproduct of `X` and `Y` exists. -/
theorem HasBinaryBiproduct.of_hasBinaryProduct (X Y : C) [HasBinaryProduct X Y] :
    HasBinaryBiproduct X Y :=
  HasBinaryBiproduct.mk
    { bicone := _
      isBilimit := binaryBiconeIsBilimitOfLimitConeOfIsLimit (limit.isLimit _) }


/-- In a preadditive category, if all binary products exist, then all binary biproducts exist. -/
theorem HasBinaryBiproducts.of_hasBinaryProducts [HasBinaryProducts C] : HasBinaryBiproducts C :=
  { has_binary_biproduct := fun X Y => HasBinaryBiproduct.of_hasBinaryProduct X Y }


/-- We can turn any colimit cocone over a pair into a bicone. -/
@[simps]
def BinaryBicone.ofColimitCocone {X Y : C} {t : Cocone (pair X Y)} (ht : IsColimit t) :
    BinaryBicone X Y where
  pt := t.pt
  fst := ht.desc (BinaryCofan.mk (𝟙 X) 0)
  snd := ht.desc (BinaryCofan.mk 0 (𝟙 Y))
  inl := t.ι.app ⟨WalkingPair.left⟩
  inr := t.ι.app ⟨WalkingPair.right⟩


theorem fst_of_isColimit {X Y : C} {t : BinaryBicone X Y} (ht : IsColimit t.toCocone) :
    t.fst = ht.desc (BinaryCofan.mk (𝟙 X) 0) := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    X Y : C
    t : CategoryTheory.Limits.BinaryBicone X Y
    ht : CategoryTheory.Limits.IsColimit t.toCocone
    ⊢ Eq t.fst (ht.desc (CategoryTheory.Limits.BinaryCofan.mk (CategoryTheory.Cate …
  -/
  apply ht.uniq (BinaryCofan.mk (𝟙 X) 0)
  /-
    case x
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    X Y : C
    t : CategoryTheory.Limits.BinaryBicone X Y
    ht : CategoryTheory.Limits.IsColimit t.toCocone
    ⊢ ∀ (j : CategoryTheory.Discrete CategoryTheory.Limits.WalkingPair), Eq (Categ …
  -/
                            /-
                              🎉 no goals
                            -/
  rintro ⟨⟨⟩⟩ <;> dsimp <;> simp
                            /-
                              🎉 no goals
                            -/


theorem snd_of_isColimit {X Y : C} {t : BinaryBicone X Y} (ht : IsColimit t.toCocone) :
    t.snd = ht.desc (BinaryCofan.mk 0 (𝟙 Y)) := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    X Y : C
    t : CategoryTheory.Limits.BinaryBicone X Y
    ht : CategoryTheory.Limits.IsColimit t.toCocone
    ⊢ Eq t.snd (ht.desc (CategoryTheory.Limits.BinaryCofan.mk 0 (CategoryTheory.Ca …
  -/
  apply ht.uniq (BinaryCofan.mk 0 (𝟙 Y))
  /-
    case x
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    X Y : C
    t : CategoryTheory.Limits.BinaryBicone X Y
    ht : CategoryTheory.Limits.IsColimit t.toCocone
    ⊢ ∀ (j : CategoryTheory.Discrete CategoryTheory.Limits.WalkingPair), Eq (Categ …
  -/
                            /-
                              🎉 no goals
                            -/
  rintro ⟨⟨⟩⟩ <;> dsimp <;> simp
                            /-
                              🎉 no goals
                            -/


/-- In a preadditive category, any binary bicone which is a colimit cocone is in fact a
    bilimit bicone. -/
def isBinaryBilimitOfIsColimit {X Y : C} (t : BinaryBicone X Y) (ht : IsColimit t.toCocone) :
    t.IsBilimit :=
  isBinaryBilimitOfTotal _ <| by
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.Preadditive C
      X Y : C
      t : CategoryTheory.Limits.BinaryBicone X Y
      ht : CategoryTheory.Limits.IsColimit t.toCocone
      ⊢ Eq (HAdd.hAdd (CategoryTheory.CategoryStruct.comp t.fst t.inl) (CategoryTheo …
    -/
                                                      /-
                                                        🎉 no goals
                                                      -/
    refine BinaryCofan.IsColimit.hom_ext ht ?_ ?_ <;> simp
                                                      /-
                                                        🎉 no goals
                                                      -/


/-- We can turn any colimit cocone over a pair into a bilimit bicone. -/
def binaryBiconeIsBilimitOfColimitCoconeOfIsColimit {X Y : C} {t : Cocone (pair X Y)}
    (ht : IsColimit t) : (BinaryBicone.ofColimitCocone ht).IsBilimit :=
  isBinaryBilimitOfIsColimit (BinaryBicone.ofColimitCocone ht) <|
    IsColimit.ofIsoColimit ht <|
      Cocones.ext (Iso.refl _) fun j => by
        /-
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          inst✝ : CategoryTheory.Preadditive C
          X Y : C
          t : CategoryTheory.Limits.Cocone (CategoryTheory.Limits.pair X Y)
          ht : CategoryTheory.Limits.IsColimit t
          j : CategoryTheory.Discrete CategoryTheory.Limits.WalkingPair
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (t.ι.app j) (CategoryTheory.Iso.refl  …
        -/
                               /-
                                 🎉 no goals
                               -/
        rcases j with ⟨⟨⟩⟩ <;> simp
                               /-
                                 🎉 no goals
                               -/


/-- In a preadditive category, if the coproduct of `X` and `Y` exists, then the
    binary biproduct of `X` and `Y` exists. -/
theorem HasBinaryBiproduct.of_hasBinaryCoproduct (X Y : C) [HasBinaryCoproduct X Y] :
    HasBinaryBiproduct X Y :=
  HasBinaryBiproduct.mk
    { bicone := _
      isBilimit := binaryBiconeIsBilimitOfColimitCoconeOfIsColimit (colimit.isColimit _) }


/-- In a preadditive category, if all binary coproducts exist, then all binary biproducts exist. -/
theorem HasBinaryBiproducts.of_hasBinaryCoproducts [HasBinaryCoproducts C] :
    HasBinaryBiproducts C :=
  { has_binary_biproduct := fun X Y => HasBinaryBiproduct.of_hasBinaryCoproduct X Y }


/-- In any preadditive category, any binary biproduct satisfies
`biprod.fst ≫ biprod.inl + biprod.snd ≫ biprod.inr = 𝟙 (X ⊞ Y)`.
-/
@[simp]
theorem biprod.total : biprod.fst ≫ biprod.inl + biprod.snd ≫ biprod.inr = 𝟙 (X ⊞ Y) := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Preadditive C
    X Y : C
    inst✝ : CategoryTheory.Limits.HasBinaryBiproduct X Y
    ⊢ Eq (HAdd.hAdd (CategoryTheory.CategoryStruct.comp CategoryTheory.Limits.bipr …
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
  ext <;> simp [add_comp]
          /-
            🎉 no goals
          -/


theorem biprod.lift_eq {T : C} {f : T ⟶ X} {g : T ⟶ Y} :
                                                            /-
                                                              C : Type u
                                                              inst✝² : CategoryTheory.Category.{v, u} C
                                                              inst✝¹ : CategoryTheory.Preadditive C
                                                              X Y : C
                                                              inst✝ : CategoryTheory.Limits.HasBinaryBiproduct X Y
                                                              T : C
                                                              f : Quiver.Hom T X
                                                              g : Quiver.Hom T Y
                                                              ⊢ Eq (CategoryTheory.Limits.biprod.lift f g) (HAdd.hAdd (CategoryTheory.Catego …
                                                            -/
                                                                    /-
                                                                      🎉 no goals
                                                                    -/
    biprod.lift f g = f ≫ biprod.inl + g ≫ biprod.inr := by ext <;> simp [add_comp]
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


theorem biprod.desc_eq {T : C} {f : X ⟶ T} {g : Y ⟶ T} :
                                                            /-
                                                              C : Type u
                                                              inst✝² : CategoryTheory.Category.{v, u} C
                                                              inst✝¹ : CategoryTheory.Preadditive C
                                                              X Y : C
                                                              inst✝ : CategoryTheory.Limits.HasBinaryBiproduct X Y
                                                              T : C
                                                              f : Quiver.Hom X T
                                                              g : Quiver.Hom Y T
                                                              ⊢ Eq (CategoryTheory.Limits.biprod.desc f g) (HAdd.hAdd (CategoryTheory.Catego …
                                                            -/
                                                                    /-
                                                                      🎉 no goals
                                                                    -/
    biprod.desc f g = biprod.fst ≫ f + biprod.snd ≫ g := by ext <;> simp [add_comp]
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


@[reassoc (attr := simp)]
theorem biprod.lift_desc {T U : C} {f : T ⟶ X} {g : T ⟶ Y} {h : X ⟶ U} {i : Y ⟶ U} :
                                                            /-
                                                              C : Type u
                                                              inst✝² : CategoryTheory.Category.{v, u} C
                                                              inst✝¹ : CategoryTheory.Preadditive C
                                                              X Y : C
                                                              inst✝ : CategoryTheory.Limits.HasBinaryBiproduct X Y
                                                              T U : C
                                                              f : Quiver.Hom T X
                                                              g : Quiver.Hom T Y
                                                              h : Quiver.Hom X U
                                                              i : Quiver.Hom Y U
                                                              ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.biprod.lift f  …
                                                            -/
    biprod.lift f g ≫ biprod.desc h i = f ≫ h + g ≫ i := by simp [biprod.lift_eq, biprod.desc_eq]
                                                            /-
                                                              🎉 no goals
                                                            -/


theorem biprod.map_eq [HasBinaryBiproducts C] {W X Y Z : C} {f : W ⟶ Y} {g : X ⟶ Z} :
    biprod.map f g = biprod.fst ≫ f ≫ biprod.inl + biprod.snd ≫ g ≫ biprod.inr := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : CategoryTheory.Limits.HasBinaryBiproducts C
    W X Y Z : C
    f : Quiver.Hom W Y
    g : Quiver.Hom X Z
    ⊢ Eq (CategoryTheory.Limits.biprod.map f g) (HAdd.hAdd (CategoryTheory.Categor …
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
  ext <;> simp
          /-
            🎉 no goals
          -/


/-- Every split mono `f` with a cokernel induces a binary bicone with `f` as its `inl` and
the cokernel map as its `snd`.
We will show in `is_bilimit_binary_bicone_of_split_mono_of_cokernel` that this binary bicone is in
fact already a biproduct. -/
@[simps]
def binaryBiconeOfIsSplitMonoOfCokernel {X Y : C} {f : X ⟶ Y} [IsSplitMono f] {c : CokernelCofork f}
    (i : IsColimit c) : BinaryBicone X c.pt where
  pt := Y
  fst := retraction f
  snd := c.π
  inl := f
  inr :=
    let c' : CokernelCofork (𝟙 Y - (𝟙 Y - retraction f ≫ f)) :=
                                          /-
                                            C : Type u
                                            inst✝³ : CategoryTheory.Category.{v, u} C
                                            inst✝² : CategoryTheory.Preadditive C
                                            X✝ Y✝ : C
                                            inst✝¹ : CategoryTheory.Limits.HasBinaryBiproduct X✝ Y✝
                                            X Y : C
                                            f : Quiver.Hom X Y
                                            inst✝ : CategoryTheory.IsSplitMono f
                                            c : CategoryTheory.Limits.CokernelCofork f
                                            i : CategoryTheory.Limits.IsColimit c
                                            ⊢ Eq (CategoryTheory.CategoryStruct.comp (HSub.hSub (CategoryTheory.CategorySt …
                                          -/
      CokernelCofork.ofπ (Cofork.π c) (by simp)
                                          /-
                                            🎉 no goals
                                          -/
                                                                    /-
                                                                      C : Type u
                                                                      inst✝³ : CategoryTheory.Category.{v, u} C
                                                                      inst✝² : CategoryTheory.Preadditive C
                                                                      X✝ Y✝ : C
                                                                      inst✝¹ : CategoryTheory.Limits.HasBinaryBiproduct X✝ Y✝
                                                                      X Y : C
                                                                      f : Quiver.Hom X Y
                                                                      inst✝ : CategoryTheory.IsSplitMono f
                                                                      c : CategoryTheory.Limits.CokernelCofork f
                                                                      i : CategoryTheory.Limits.IsColimit c
                                                                      c' : CategoryTheory.Limits.CokernelCofork (HSub.hSub (CategoryTheory.CategoryS …
                                                                      ⊢ Eq (HSub.hSub (CategoryTheory.CategoryStruct.id Y) (HSub.hSub (CategoryTheor …
                                                                    -/
    let i' : IsColimit c' := isCokernelEpiComp i (retraction f) (by simp)
                                                                    /-
                                                                      🎉 no goals
                                                                    -/
    let i'' := isColimitCoforkOfCokernelCofork i'
                                                 /-
                                                   C : Type u
                                                   inst✝³ : CategoryTheory.Category.{v, u} C
                                                   inst✝² : CategoryTheory.Preadditive C
                                                   X✝ Y✝ : C
                                                   inst✝¹ : CategoryTheory.Limits.HasBinaryBiproduct X✝ Y✝
                                                   X Y : C
                                                   f : Quiver.Hom X Y
                                                   inst✝ : CategoryTheory.IsSplitMono f
                                                   c : CategoryTheory.Limits.CokernelCofork f
                                                   i : CategoryTheory.Limits.IsColimit c
                                                   c' : CategoryTheory.Limits.CokernelCofork (HSub.hSub (CategoryTheory.CategoryS …
                                                   i' : CategoryTheory.Limits.IsColimit c' := CategoryTheory.Limits.isCokernelEpi …
                                                   i'' : CategoryTheory.Limits.IsColimit (CategoryTheory.Preadditive.coforkOfCoke …
                                                   ⊢ Eq (CategoryTheory.CategoryStruct.comp (HSub.hSub (CategoryTheory.CategorySt …
                                                 -/
    (splitEpiOfIdempotentOfIsColimitCofork C (by simp) i'').section_
                                                 /-
                                                   🎉 no goals
                                                 -/
                /-
                  C : Type u
                  inst✝³ : CategoryTheory.Category.{v, u} C
                  inst✝² : CategoryTheory.Preadditive C
                  X✝ Y✝ : C
                  inst✝¹ : CategoryTheory.Limits.HasBinaryBiproduct X✝ Y✝
                  X Y : C
                  f : Quiver.Hom X Y
                  inst✝ : CategoryTheory.IsSplitMono f
                  c : CategoryTheory.Limits.CokernelCofork f
                  i : CategoryTheory.Limits.IsColimit c
                  ⊢ Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.retraction f)) (Cat …
                -/
  inl_fst := by simp
                /-
                  🎉 no goals
                -/
                /-
                  C : Type u
                  inst✝³ : CategoryTheory.Category.{v, u} C
                  inst✝² : CategoryTheory.Preadditive C
                  X✝ Y✝ : C
                  inst✝¹ : CategoryTheory.Limits.HasBinaryBiproduct X✝ Y✝
                  X Y : C
                  f : Quiver.Hom X Y
                  inst✝ : CategoryTheory.IsSplitMono f
                  c : CategoryTheory.Limits.CokernelCofork f
                  i : CategoryTheory.Limits.IsColimit c
                  ⊢ Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.Limits.Cofork.π c)) 0
                -/
  inl_snd := by simp
                /-
                  🎉 no goals
                -/
  inr_fst := by
    /-
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.Preadditive C
      X✝ Y✝ : C
      inst✝¹ : CategoryTheory.Limits.HasBinaryBiproduct X✝ Y✝
      X Y : C
      f : Quiver.Hom X Y
      inst✝ : CategoryTheory.IsSplitMono f
      c : CategoryTheory.Limits.CokernelCofork f
      i : CategoryTheory.Limits.IsColimit c
      ⊢ Eq
          (CategoryTheory.CategoryStruct.comp
            (let c' := CategoryTheory.Limits.CokernelCofork.ofπ (CategoryTheory.Limi …
            let i' := CategoryTheory.Limits.isCokernelEpiComp i (CategoryTheory.retr …
            let i'' := CategoryTheory.Preadditive.isColimitCoforkOfCokernelCofork i';
            (CategoryTheory.Limits.splitEpiOfIdempotentOfIsColimitCofork C ⋯ i'').se …
            (CategoryTheory.retraction f))
          0
    -/
    dsimp only
    rw [splitEpiOfIdempotentOfIsColimitCofork_section_,
      isColimitCoforkOfCokernelCofork_desc, isCokernelEpiComp_desc]
    /-
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.Preadditive C
      X✝ Y✝ : C
      inst✝¹ : CategoryTheory.Limits.HasBinaryBiproduct X✝ Y✝
      X Y : C
      f : Quiver.Hom X Y
      inst✝ : CategoryTheory.IsSplitMono f
      c : CategoryTheory.Limits.CokernelCofork f
      i : CategoryTheory.Limits.IsColimit c
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (i.desc (CategoryTheory.Limits.Cofork …
    -/
    dsimp only [cokernelCoforkOfCofork_ofπ]
    /-
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.Preadditive C
      X✝ Y✝ : C
      inst✝¹ : CategoryTheory.Limits.HasBinaryBiproduct X✝ Y✝
      X Y : C
      f : Quiver.Hom X Y
      inst✝ : CategoryTheory.IsSplitMono f
      c : CategoryTheory.Limits.CokernelCofork f
      i : CategoryTheory.Limits.IsColimit c
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (i.desc (CategoryTheory.Limits.Cofork …
    -/
    letI := epi_of_isColimit_cofork i
    /-
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.Preadditive C
      X✝ Y✝ : C
      inst✝¹ : CategoryTheory.Limits.HasBinaryBiproduct X✝ Y✝
      X Y : C
      f : Quiver.Hom X Y
      inst✝ : CategoryTheory.IsSplitMono f
      c : CategoryTheory.Limits.CokernelCofork f
      i : CategoryTheory.Limits.IsColimit c
      this : CategoryTheory.Epi (CategoryTheory.Limits.Cofork.π c) := CategoryTheory …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (i.desc (CategoryTheory.Limits.Cofork …
    -/
    apply zero_of_epi_comp c.π
    simp only [sub_comp, comp_sub, Category.comp_id, Category.assoc, IsSplitMono.id, sub_self,
      Cofork.IsColimit.π_desc_assoc, CokernelCofork.π_ofπ, IsSplitMono.id_assoc]
    /-
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.Preadditive C
      X✝ Y✝ : C
      inst✝¹ : CategoryTheory.Limits.HasBinaryBiproduct X✝ Y✝
      X Y : C
      f : Quiver.Hom X Y
      inst✝ : CategoryTheory.IsSplitMono f
      c : CategoryTheory.Limits.CokernelCofork f
      i : CategoryTheory.Limits.IsColimit c
      this : CategoryTheory.Epi (CategoryTheory.Limits.Cofork.π c) := CategoryTheory …
      ⊢ Eq (HSub.hSub (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategorySt …
    -/
    apply sub_eq_zero_of_eq
    /-
      case a
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.Preadditive C
      X✝ Y✝ : C
      inst✝¹ : CategoryTheory.Limits.HasBinaryBiproduct X✝ Y✝
      X Y : C
      f : Quiver.Hom X Y
      inst✝ : CategoryTheory.IsSplitMono f
      c : CategoryTheory.Limits.CokernelCofork f
      i : CategoryTheory.Limits.IsColimit c
      this : CategoryTheory.Epi (CategoryTheory.Limits.Cofork.π c) := CategoryTheory …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id Y)  …
    -/
    apply Category.id_comp
    /-
      🎉 no goals
    -/
                /-
                  C : Type u
                  inst✝³ : CategoryTheory.Category.{v, u} C
                  inst✝² : CategoryTheory.Preadditive C
                  X✝ Y✝ : C
                  inst✝¹ : CategoryTheory.Limits.HasBinaryBiproduct X✝ Y✝
                  X Y : C
                  f : Quiver.Hom X Y
                  inst✝ : CategoryTheory.IsSplitMono f
                  c : CategoryTheory.Limits.CokernelCofork f
                  i : CategoryTheory.Limits.IsColimit c
                  ⊢ Eq
                      (CategoryTheory.CategoryStruct.comp
                        (let c' := CategoryTheory.Limits.CokernelCofork.ofπ (CategoryTheory.Limi …
                        let i' := CategoryTheory.Limits.isCokernelEpiComp i (CategoryTheory.retr …
                        let i'' := CategoryTheory.Preadditive.isColimitCoforkOfCokernelCofork i';
                        (CategoryTheory.Limits.splitEpiOfIdempotentOfIsColimitCofork C ⋯ i'').se …
                        (CategoryTheory.Limits.Cofork.π c))
                      (CategoryTheory.CategoryStruct.id c.pt)
                -/
  inr_snd := by apply SplitEpi.id
                /-
                  🎉 no goals
                -/


/-- The bicone constructed in `binaryBiconeOfSplitMonoOfCokernel` is a bilimit.
This is a version of the splitting lemma that holds in all preadditive categories. -/
def isBilimitBinaryBiconeOfIsSplitMonoOfCokernel {X Y : C} {f : X ⟶ Y} [IsSplitMono f]
    {c : CokernelCofork f} (i : IsColimit c) : (binaryBiconeOfIsSplitMonoOfCokernel i).IsBilimit :=
  isBinaryBilimitOfTotal _
    (by
      simp only [binaryBiconeOfIsSplitMonoOfCokernel_fst,
        binaryBiconeOfIsSplitMonoOfCokernel_inr,
        binaryBiconeOfIsSplitMonoOfCokernel_snd,
        splitEpiOfIdempotentOfIsColimitCofork_section_]
      /-
        C : Type u
        inst✝³ : CategoryTheory.Category.{v, u} C
        inst✝² : CategoryTheory.Preadditive C
        X✝ Y✝ : C
        inst✝¹ : CategoryTheory.Limits.HasBinaryBiproduct X✝ Y✝
        X Y : C
        f : Quiver.Hom X Y
        inst✝ : CategoryTheory.IsSplitMono f
        c : CategoryTheory.Limits.CokernelCofork f
        i : CategoryTheory.Limits.IsColimit c
        ⊢ Eq (HAdd.hAdd (CategoryTheory.CategoryStruct.comp (CategoryTheory.retraction …
      -/
      dsimp only [binaryBiconeOfIsSplitMonoOfCokernel_pt]
      /-
        C : Type u
        inst✝³ : CategoryTheory.Category.{v, u} C
        inst✝² : CategoryTheory.Preadditive C
        X✝ Y✝ : C
        inst✝¹ : CategoryTheory.Limits.HasBinaryBiproduct X✝ Y✝
        X Y : C
        f : Quiver.Hom X Y
        inst✝ : CategoryTheory.IsSplitMono f
        c : CategoryTheory.Limits.CokernelCofork f
        i : CategoryTheory.Limits.IsColimit c
        ⊢ Eq (HAdd.hAdd (CategoryTheory.CategoryStruct.comp (CategoryTheory.retraction …
      -/
      rw [isColimitCoforkOfCokernelCofork_desc, isCokernelEpiComp_desc]
      simp only [binaryBiconeOfIsSplitMonoOfCokernel_inl, Cofork.IsColimit.π_desc,
        cokernelCoforkOfCofork_π, Cofork.π_ofπ, add_sub_cancel])


/-- If `b` is a binary bicone such that `b.inl` is a kernel of `b.snd`, then `b` is a bilimit
    bicone. -/
def BinaryBicone.isBilimitOfKernelInl {X Y : C} (b : BinaryBicone X Y)
    (hb : IsLimit b.sndKernelFork) : b.IsBilimit :=
  isBinaryBilimitOfIsLimit _ <|
                                                                             /-
                                                                               C : Type u
                                                                               inst✝² : CategoryTheory.Category.{v, u} C
                                                                               inst✝¹ : CategoryTheory.Preadditive C
                                                                               X✝ Y✝ : C
                                                                               inst✝ : CategoryTheory.Limits.HasBinaryBiproduct X✝ Y✝
                                                                               X Y : C
                                                                               b : CategoryTheory.Limits.BinaryBicone X Y
                                                                               hb : CategoryTheory.Limits.IsLimit b.sndKernelFork
                                                                               T✝ : C
                                                                               f : Quiver.Hom T✝ X
                                                                               g : Quiver.Hom T✝ Y
                                                                               ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun {T} f g => HAdd.hAdd (CategoryT …
                                                                             -/
    BinaryFan.IsLimit.mk _ (fun f g => f ≫ b.inl + g ≫ b.inr) (fun f g => by simp)
                                                                             /-
                                                                               🎉 no goals
                                                                             -/
                     /-
                       C : Type u
                       inst✝² : CategoryTheory.Category.{v, u} C
                       inst✝¹ : CategoryTheory.Preadditive C
                       X✝ Y✝ : C
                       inst✝ : CategoryTheory.Limits.HasBinaryBiproduct X✝ Y✝
                       X Y : C
                       b : CategoryTheory.Limits.BinaryBicone X Y
                       hb : CategoryTheory.Limits.IsLimit b.sndKernelFork
                       T✝ : C
                       f : Quiver.Hom T✝ X
                       g : Quiver.Hom T✝ Y
                       ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun {T} f g => HAdd.hAdd (CategoryT …
                     -/
      (fun f g => by simp) fun {T} f g m h₁ h₂ => by
                     /-
                       🎉 no goals
                     -/
      /-
        C : Type u
        inst✝² : CategoryTheory.Category.{v, u} C
        inst✝¹ : CategoryTheory.Preadditive C
        X✝ Y✝ : C
        inst✝ : CategoryTheory.Limits.HasBinaryBiproduct X✝ Y✝
        X Y : C
        b : CategoryTheory.Limits.BinaryBicone X Y
        hb : CategoryTheory.Limits.IsLimit b.sndKernelFork
        T : C
        f : Quiver.Hom T X
        g : Quiver.Hom T Y
        m : Quiver.Hom T b.toCone.pt
        h₁ : Eq (CategoryTheory.CategoryStruct.comp m (CategoryTheory.Limits.BinaryFan …
        h₂ : Eq (CategoryTheory.CategoryStruct.comp m (CategoryTheory.Limits.BinaryFan …
        ⊢ Eq m ((fun {T} f g => HAdd.hAdd (CategoryTheory.CategoryStruct.comp f b.inl) …
      -/
      dsimp at m
      have h₁' : ((m : T ⟶ b.pt) - (f ≫ b.inl + g ≫ b.inr)) ≫ b.fst = 0 := by
        simpa using sub_eq_zero.2 h₁
      /-
        C : Type u
        inst✝² : CategoryTheory.Category.{v, u} C
        inst✝¹ : CategoryTheory.Preadditive C
        X✝ Y✝ : C
        inst✝ : CategoryTheory.Limits.HasBinaryBiproduct X✝ Y✝
        X Y : C
        b : CategoryTheory.Limits.BinaryBicone X Y
        hb : CategoryTheory.Limits.IsLimit b.sndKernelFork
        T : C
        f : Quiver.Hom T X
        g : Quiver.Hom T Y
        m : Quiver.Hom T b.pt
        h₁ : Eq (CategoryTheory.CategoryStruct.comp m (CategoryTheory.Limits.BinaryFan …
        h₂ : Eq (CategoryTheory.CategoryStruct.comp m (CategoryTheory.Limits.BinaryFan …
        h₁' : Eq (CategoryTheory.CategoryStruct.comp (HSub.hSub m (HAdd.hAdd (Category …
        ⊢ Eq m ((fun {T} f g => HAdd.hAdd (CategoryTheory.CategoryStruct.comp f b.inl) …
      -/
      have h₂' : (m - (f ≫ b.inl + g ≫ b.inr)) ≫ b.snd = 0 := by simpa using sub_eq_zero.2 h₂
      obtain ⟨q : T ⟶ X, hq : q ≫ b.inl = m - (f ≫ b.inl + g ≫ b.inr)⟩ :=
        KernelFork.IsLimit.lift' hb _ h₂'
      rw [← sub_eq_zero, ← hq, ← Category.comp_id q, ← b.inl_fst, ← Category.assoc, hq, h₁',
        zero_comp]


/-- If `b` is a binary bicone such that `b.inr` is a kernel of `b.fst`, then `b` is a bilimit
    bicone. -/
def BinaryBicone.isBilimitOfKernelInr {X Y : C} (b : BinaryBicone X Y)
    (hb : IsLimit b.fstKernelFork) : b.IsBilimit :=
  isBinaryBilimitOfIsLimit _ <|
                                                                             /-
                                                                               C : Type u
                                                                               inst✝² : CategoryTheory.Category.{v, u} C
                                                                               inst✝¹ : CategoryTheory.Preadditive C
                                                                               X✝ Y✝ : C
                                                                               inst✝ : CategoryTheory.Limits.HasBinaryBiproduct X✝ Y✝
                                                                               X Y : C
                                                                               b : CategoryTheory.Limits.BinaryBicone X Y
                                                                               hb : CategoryTheory.Limits.IsLimit b.fstKernelFork
                                                                               T✝ : C
                                                                               f : Quiver.Hom T✝ X
                                                                               g : Quiver.Hom T✝ Y
                                                                               ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun {T} f g => HAdd.hAdd (CategoryT …
                                                                             -/
    BinaryFan.IsLimit.mk _ (fun f g => f ≫ b.inl + g ≫ b.inr) (fun f g => by simp)
                                                                             /-
                                                                               🎉 no goals
                                                                             -/
                   /-
                     C : Type u
                     inst✝² : CategoryTheory.Category.{v, u} C
                     inst✝¹ : CategoryTheory.Preadditive C
                     X✝ Y✝ : C
                     inst✝ : CategoryTheory.Limits.HasBinaryBiproduct X✝ Y✝
                     X Y : C
                     b : CategoryTheory.Limits.BinaryBicone X Y
                     hb : CategoryTheory.Limits.IsLimit b.fstKernelFork
                     T✝ : C
                     f : Quiver.Hom T✝ X
                     g : Quiver.Hom T✝ Y
                     ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun {T} f g => HAdd.hAdd (CategoryT …
                   -/
    (fun f g => by simp) fun {T} f g m h₁ h₂ => by
                   /-
                     🎉 no goals
                   -/
      /-
        C : Type u
        inst✝² : CategoryTheory.Category.{v, u} C
        inst✝¹ : CategoryTheory.Preadditive C
        X✝ Y✝ : C
        inst✝ : CategoryTheory.Limits.HasBinaryBiproduct X✝ Y✝
        X Y : C
        b : CategoryTheory.Limits.BinaryBicone X Y
        hb : CategoryTheory.Limits.IsLimit b.fstKernelFork
        T : C
        f : Quiver.Hom T X
        g : Quiver.Hom T Y
        m : Quiver.Hom T b.toCone.pt
        h₁ : Eq (CategoryTheory.CategoryStruct.comp m (CategoryTheory.Limits.BinaryFan …
        h₂ : Eq (CategoryTheory.CategoryStruct.comp m (CategoryTheory.Limits.BinaryFan …
        ⊢ Eq m ((fun {T} f g => HAdd.hAdd (CategoryTheory.CategoryStruct.comp f b.inl) …
      -/
      dsimp at m
      /-
        C : Type u
        inst✝² : CategoryTheory.Category.{v, u} C
        inst✝¹ : CategoryTheory.Preadditive C
        X✝ Y✝ : C
        inst✝ : CategoryTheory.Limits.HasBinaryBiproduct X✝ Y✝
        X Y : C
        b : CategoryTheory.Limits.BinaryBicone X Y
        hb : CategoryTheory.Limits.IsLimit b.fstKernelFork
        T : C
        f : Quiver.Hom T X
        g : Quiver.Hom T Y
        m : Quiver.Hom T b.pt
        h₁ : Eq (CategoryTheory.CategoryStruct.comp m (CategoryTheory.Limits.BinaryFan …
        h₂ : Eq (CategoryTheory.CategoryStruct.comp m (CategoryTheory.Limits.BinaryFan …
        ⊢ Eq m ((fun {T} f g => HAdd.hAdd (CategoryTheory.CategoryStruct.comp f b.inl) …
      -/
      have h₁' : (m - (f ≫ b.inl + g ≫ b.inr)) ≫ b.fst = 0 := by simpa using sub_eq_zero.2 h₁
      /-
        C : Type u
        inst✝² : CategoryTheory.Category.{v, u} C
        inst✝¹ : CategoryTheory.Preadditive C
        X✝ Y✝ : C
        inst✝ : CategoryTheory.Limits.HasBinaryBiproduct X✝ Y✝
        X Y : C
        b : CategoryTheory.Limits.BinaryBicone X Y
        hb : CategoryTheory.Limits.IsLimit b.fstKernelFork
        T : C
        f : Quiver.Hom T X
        g : Quiver.Hom T Y
        m : Quiver.Hom T b.pt
        h₁ : Eq (CategoryTheory.CategoryStruct.comp m (CategoryTheory.Limits.BinaryFan …
        h₂ : Eq (CategoryTheory.CategoryStruct.comp m (CategoryTheory.Limits.BinaryFan …
        h₁' : Eq (CategoryTheory.CategoryStruct.comp (HSub.hSub m (HAdd.hAdd (Category …
        ⊢ Eq m ((fun {T} f g => HAdd.hAdd (CategoryTheory.CategoryStruct.comp f b.inl) …
      -/
      have h₂' : (m - (f ≫ b.inl + g ≫ b.inr)) ≫ b.snd = 0 := by simpa using sub_eq_zero.2 h₂
      obtain ⟨q : T ⟶ Y, hq : q ≫ b.inr = m - (f ≫ b.inl + g ≫ b.inr)⟩ :=
        KernelFork.IsLimit.lift' hb _ h₁'
      rw [← sub_eq_zero, ← hq, ← Category.comp_id q, ← b.inr_snd, ← Category.assoc, hq, h₂',
        zero_comp]


/-- If `b` is a binary bicone such that `b.fst` is a cokernel of `b.inr`, then `b` is a bilimit
    bicone. -/
def BinaryBicone.isBilimitOfCokernelFst {X Y : C} (b : BinaryBicone X Y)
    (hb : IsColimit b.inrCokernelCofork) : b.IsBilimit :=
  isBinaryBilimitOfIsColimit _ <|
                                                                                 /-
                                                                                   C : Type u
                                                                                   inst✝² : CategoryTheory.Category.{v, u} C
                                                                                   inst✝¹ : CategoryTheory.Preadditive C
                                                                                   X✝ Y✝ : C
                                                                                   inst✝ : CategoryTheory.Limits.HasBinaryBiproduct X✝ Y✝
                                                                                   X Y : C
                                                                                   b : CategoryTheory.Limits.BinaryBicone X Y
                                                                                   hb : CategoryTheory.Limits.IsColimit b.inrCokernelCofork
                                                                                   T✝ : C
                                                                                   f : Quiver.Hom X T✝
                                                                                   g : Quiver.Hom Y T✝
                                                                                   ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.BinaryCofan.in …
                                                                                 -/
    BinaryCofan.IsColimit.mk _ (fun f g => b.fst ≫ f + b.snd ≫ g) (fun f g => by simp)
                                                                                 /-
                                                                                   🎉 no goals
                                                                                 -/
                     /-
                       C : Type u
                       inst✝² : CategoryTheory.Category.{v, u} C
                       inst✝¹ : CategoryTheory.Preadditive C
                       X✝ Y✝ : C
                       inst✝ : CategoryTheory.Limits.HasBinaryBiproduct X✝ Y✝
                       X Y : C
                       b : CategoryTheory.Limits.BinaryBicone X Y
                       hb : CategoryTheory.Limits.IsColimit b.inrCokernelCofork
                       T✝ : C
                       f : Quiver.Hom X T✝
                       g : Quiver.Hom Y T✝
                       ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.BinaryCofan.in …
                     -/
      (fun f g => by simp) fun {T} f g m h₁ h₂ => by
                     /-
                       🎉 no goals
                     -/
      /-
        C : Type u
        inst✝² : CategoryTheory.Category.{v, u} C
        inst✝¹ : CategoryTheory.Preadditive C
        X✝ Y✝ : C
        inst✝ : CategoryTheory.Limits.HasBinaryBiproduct X✝ Y✝
        X Y : C
        b : CategoryTheory.Limits.BinaryBicone X Y
        hb : CategoryTheory.Limits.IsColimit b.inrCokernelCofork
        T : C
        f : Quiver.Hom X T
        g : Quiver.Hom Y T
        m : Quiver.Hom b.toCocone.pt T
        h₁ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.BinaryCofan …
        h₂ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.BinaryCofan …
        ⊢ Eq m ((fun {T} f g => HAdd.hAdd (CategoryTheory.CategoryStruct.comp b.fst f) …
      -/
      dsimp at m
      /-
        C : Type u
        inst✝² : CategoryTheory.Category.{v, u} C
        inst✝¹ : CategoryTheory.Preadditive C
        X✝ Y✝ : C
        inst✝ : CategoryTheory.Limits.HasBinaryBiproduct X✝ Y✝
        X Y : C
        b : CategoryTheory.Limits.BinaryBicone X Y
        hb : CategoryTheory.Limits.IsColimit b.inrCokernelCofork
        T : C
        f : Quiver.Hom X T
        g : Quiver.Hom Y T
        m : Quiver.Hom b.pt T
        h₁ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.BinaryCofan …
        h₂ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.BinaryCofan …
        ⊢ Eq m ((fun {T} f g => HAdd.hAdd (CategoryTheory.CategoryStruct.comp b.fst f) …
      -/
      have h₁' : b.inl ≫ (m - (b.fst ≫ f + b.snd ≫ g)) = 0 := by simpa using sub_eq_zero.2 h₁
      /-
        C : Type u
        inst✝² : CategoryTheory.Category.{v, u} C
        inst✝¹ : CategoryTheory.Preadditive C
        X✝ Y✝ : C
        inst✝ : CategoryTheory.Limits.HasBinaryBiproduct X✝ Y✝
        X Y : C
        b : CategoryTheory.Limits.BinaryBicone X Y
        hb : CategoryTheory.Limits.IsColimit b.inrCokernelCofork
        T : C
        f : Quiver.Hom X T
        g : Quiver.Hom Y T
        m : Quiver.Hom b.pt T
        h₁ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.BinaryCofan …
        h₂ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.BinaryCofan …
        h₁' : Eq (CategoryTheory.CategoryStruct.comp b.inl (HSub.hSub m (HAdd.hAdd (Ca …
        ⊢ Eq m ((fun {T} f g => HAdd.hAdd (CategoryTheory.CategoryStruct.comp b.fst f) …
      -/
      have h₂' : b.inr ≫ (m - (b.fst ≫ f + b.snd ≫ g)) = 0 := by simpa using sub_eq_zero.2 h₂
      obtain ⟨q : X ⟶ T, hq : b.fst ≫ q = m - (b.fst ≫ f + b.snd ≫ g)⟩ :=
        CokernelCofork.IsColimit.desc' hb _ h₂'
      rw [← sub_eq_zero, ← hq, ← Category.id_comp q, ← b.inl_fst, Category.assoc, hq, h₁',
        comp_zero]


/-- If `b` is a binary bicone such that `b.snd` is a cokernel of `b.inl`, then `b` is a bilimit
    bicone. -/
def BinaryBicone.isBilimitOfCokernelSnd {X Y : C} (b : BinaryBicone X Y)
    (hb : IsColimit b.inlCokernelCofork) : b.IsBilimit :=
  isBinaryBilimitOfIsColimit _ <|
                                                                                 /-
                                                                                   C : Type u
                                                                                   inst✝² : CategoryTheory.Category.{v, u} C
                                                                                   inst✝¹ : CategoryTheory.Preadditive C
                                                                                   X✝ Y✝ : C
                                                                                   inst✝ : CategoryTheory.Limits.HasBinaryBiproduct X✝ Y✝
                                                                                   X Y : C
                                                                                   b : CategoryTheory.Limits.BinaryBicone X Y
                                                                                   hb : CategoryTheory.Limits.IsColimit b.inlCokernelCofork
                                                                                   T✝ : C
                                                                                   f : Quiver.Hom X T✝
                                                                                   g : Quiver.Hom Y T✝
                                                                                   ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.BinaryCofan.in …
                                                                                 -/
    BinaryCofan.IsColimit.mk _ (fun f g => b.fst ≫ f + b.snd ≫ g) (fun f g => by simp)
                                                                                 /-
                                                                                   🎉 no goals
                                                                                 -/
                     /-
                       C : Type u
                       inst✝² : CategoryTheory.Category.{v, u} C
                       inst✝¹ : CategoryTheory.Preadditive C
                       X✝ Y✝ : C
                       inst✝ : CategoryTheory.Limits.HasBinaryBiproduct X✝ Y✝
                       X Y : C
                       b : CategoryTheory.Limits.BinaryBicone X Y
                       hb : CategoryTheory.Limits.IsColimit b.inlCokernelCofork
                       T✝ : C
                       f : Quiver.Hom X T✝
                       g : Quiver.Hom Y T✝
                       ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.BinaryCofan.in …
                     -/
      (fun f g => by simp) fun {T} f g m h₁ h₂ => by
                     /-
                       🎉 no goals
                     -/
      /-
        C : Type u
        inst✝² : CategoryTheory.Category.{v, u} C
        inst✝¹ : CategoryTheory.Preadditive C
        X✝ Y✝ : C
        inst✝ : CategoryTheory.Limits.HasBinaryBiproduct X✝ Y✝
        X Y : C
        b : CategoryTheory.Limits.BinaryBicone X Y
        hb : CategoryTheory.Limits.IsColimit b.inlCokernelCofork
        T : C
        f : Quiver.Hom X T
        g : Quiver.Hom Y T
        m : Quiver.Hom b.toCocone.pt T
        h₁ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.BinaryCofan …
        h₂ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.BinaryCofan …
        ⊢ Eq m ((fun {T} f g => HAdd.hAdd (CategoryTheory.CategoryStruct.comp b.fst f) …
      -/
      dsimp at m
      /-
        C : Type u
        inst✝² : CategoryTheory.Category.{v, u} C
        inst✝¹ : CategoryTheory.Preadditive C
        X✝ Y✝ : C
        inst✝ : CategoryTheory.Limits.HasBinaryBiproduct X✝ Y✝
        X Y : C
        b : CategoryTheory.Limits.BinaryBicone X Y
        hb : CategoryTheory.Limits.IsColimit b.inlCokernelCofork
        T : C
        f : Quiver.Hom X T
        g : Quiver.Hom Y T
        m : Quiver.Hom b.pt T
        h₁ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.BinaryCofan …
        h₂ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.BinaryCofan …
        ⊢ Eq m ((fun {T} f g => HAdd.hAdd (CategoryTheory.CategoryStruct.comp b.fst f) …
      -/
      have h₁' : b.inl ≫ (m - (b.fst ≫ f + b.snd ≫ g)) = 0 := by simpa using sub_eq_zero.2 h₁
      /-
        C : Type u
        inst✝² : CategoryTheory.Category.{v, u} C
        inst✝¹ : CategoryTheory.Preadditive C
        X✝ Y✝ : C
        inst✝ : CategoryTheory.Limits.HasBinaryBiproduct X✝ Y✝
        X Y : C
        b : CategoryTheory.Limits.BinaryBicone X Y
        hb : CategoryTheory.Limits.IsColimit b.inlCokernelCofork
        T : C
        f : Quiver.Hom X T
        g : Quiver.Hom Y T
        m : Quiver.Hom b.pt T
        h₁ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.BinaryCofan …
        h₂ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.BinaryCofan …
        h₁' : Eq (CategoryTheory.CategoryStruct.comp b.inl (HSub.hSub m (HAdd.hAdd (Ca …
        ⊢ Eq m ((fun {T} f g => HAdd.hAdd (CategoryTheory.CategoryStruct.comp b.fst f) …
      -/
      have h₂' : b.inr ≫ (m - (b.fst ≫ f + b.snd ≫ g)) = 0 := by simpa using sub_eq_zero.2 h₂
      obtain ⟨q : Y ⟶ T, hq : b.snd ≫ q = m - (b.fst ≫ f + b.snd ≫ g)⟩ :=
        CokernelCofork.IsColimit.desc' hb _ h₁'
      rw [← sub_eq_zero, ← hq, ← Category.id_comp q, ← b.inr_snd, Category.assoc, hq, h₂',
        comp_zero]


/-- Every split epi `f` with a kernel induces a binary bicone with `f` as its `snd` and
the kernel map as its `inl`.
We will show in `binary_bicone_of_is_split_mono_of_cokernel` that this binary bicone is in fact
already a biproduct. -/
@[simps]
def binaryBiconeOfIsSplitEpiOfKernel {X Y : C} {f : X ⟶ Y} [IsSplitEpi f] {c : KernelFork f}
    (i : IsLimit c) : BinaryBicone c.pt Y :=
  { pt := X
    fst :=
                                                                                          /-
                                                                                            C : Type u
                                                                                            inst✝³ : CategoryTheory.Category.{v, u} C
                                                                                            inst✝² : CategoryTheory.Preadditive C
                                                                                            X✝ Y✝ : C
                                                                                            inst✝¹ : CategoryTheory.Limits.HasBinaryBiproduct X✝ Y✝
                                                                                            X Y : C
                                                                                            f : Quiver.Hom X Y
                                                                                            inst✝ : CategoryTheory.IsSplitEpi f
                                                                                            c : CategoryTheory.Limits.KernelFork f
                                                                                            i : CategoryTheory.Limits.IsLimit c
                                                                                            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Fork.ι c) (HSu …
                                                                                          -/
      let c' : KernelFork (𝟙 X - (𝟙 X - f ≫ section_ f)) := KernelFork.ofι (Fork.ι c) (by simp)
                                                                                          /-
                                                                                            🎉 no goals
                                                                                          -/
                                                                 /-
                                                                   C : Type u
                                                                   inst✝³ : CategoryTheory.Category.{v, u} C
                                                                   inst✝² : CategoryTheory.Preadditive C
                                                                   X✝ Y✝ : C
                                                                   inst✝¹ : CategoryTheory.Limits.HasBinaryBiproduct X✝ Y✝
                                                                   X Y : C
                                                                   f : Quiver.Hom X Y
                                                                   inst✝ : CategoryTheory.IsSplitEpi f
                                                                   c : CategoryTheory.Limits.KernelFork f
                                                                   i : CategoryTheory.Limits.IsLimit c
                                                                   c' : CategoryTheory.Limits.KernelFork (HSub.hSub (CategoryTheory.CategoryStruc …
                                                                   ⊢ Eq (HSub.hSub (CategoryTheory.CategoryStruct.id X) (HSub.hSub (CategoryTheor …
                                                                 -/
      let i' : IsLimit c' := isKernelCompMono i (section_ f) (by simp)
                                                                 /-
                                                                   🎉 no goals
                                                                 -/
      let i'' := isLimitForkOfKernelFork i'
                                                /-
                                                  C : Type u
                                                  inst✝³ : CategoryTheory.Category.{v, u} C
                                                  inst✝² : CategoryTheory.Preadditive C
                                                  X✝ Y✝ : C
                                                  inst✝¹ : CategoryTheory.Limits.HasBinaryBiproduct X✝ Y✝
                                                  X Y : C
                                                  f : Quiver.Hom X Y
                                                  inst✝ : CategoryTheory.IsSplitEpi f
                                                  c : CategoryTheory.Limits.KernelFork f
                                                  i : CategoryTheory.Limits.IsLimit c
                                                  c' : CategoryTheory.Limits.KernelFork (HSub.hSub (CategoryTheory.CategoryStruc …
                                                  i' : CategoryTheory.Limits.IsLimit c' := CategoryTheory.Limits.isKernelCompMon …
                                                  i'' : CategoryTheory.Limits.IsLimit (CategoryTheory.Preadditive.forkOfKernelFo …
                                                  ⊢ Eq (CategoryTheory.CategoryStruct.comp (HSub.hSub (CategoryTheory.CategorySt …
                                                -/
      (splitMonoOfIdempotentOfIsLimitFork C (by simp) i'').retraction
                                                /-
                                                  🎉 no goals
                                                -/
    snd := f
    inl := c.ι
    inr := section_ f
                  /-
                    C : Type u
                    inst✝³ : CategoryTheory.Category.{v, u} C
                    inst✝² : CategoryTheory.Preadditive C
                    X✝ Y✝ : C
                    inst✝¹ : CategoryTheory.Limits.HasBinaryBiproduct X✝ Y✝
                    X Y : C
                    f : Quiver.Hom X Y
                    inst✝ : CategoryTheory.IsSplitEpi f
                    c : CategoryTheory.Limits.KernelFork f
                    i : CategoryTheory.Limits.IsLimit c
                    ⊢ Eq
                        (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Fork.ι c)
                          (let c' := CategoryTheory.Limits.KernelFork.ofι (CategoryTheory.Limits.F …
                          let i' := CategoryTheory.Limits.isKernelCompMono i (CategoryTheory.secti …
                          let i'' := CategoryTheory.Preadditive.isLimitForkOfKernelFork i';
                          (CategoryTheory.Limits.splitMonoOfIdempotentOfIsLimitFork C ⋯ i'').retra …
                        (CategoryTheory.CategoryStruct.id c.pt)
                  -/
    inl_fst := by apply SplitMono.id
                  /-
                    🎉 no goals
                  -/
                  /-
                    C : Type u
                    inst✝³ : CategoryTheory.Category.{v, u} C
                    inst✝² : CategoryTheory.Preadditive C
                    X✝ Y✝ : C
                    inst✝¹ : CategoryTheory.Limits.HasBinaryBiproduct X✝ Y✝
                    X Y : C
                    f : Quiver.Hom X Y
                    inst✝ : CategoryTheory.IsSplitEpi f
                    c : CategoryTheory.Limits.KernelFork f
                    i : CategoryTheory.Limits.IsLimit c
                    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Fork.ι c) f) 0
                  -/
    inl_snd := by simp
                  /-
                    🎉 no goals
                  -/
    inr_fst := by
      /-
        C : Type u
        inst✝³ : CategoryTheory.Category.{v, u} C
        inst✝² : CategoryTheory.Preadditive C
        X✝ Y✝ : C
        inst✝¹ : CategoryTheory.Limits.HasBinaryBiproduct X✝ Y✝
        X Y : C
        f : Quiver.Hom X Y
        inst✝ : CategoryTheory.IsSplitEpi f
        c : CategoryTheory.Limits.KernelFork f
        i : CategoryTheory.Limits.IsLimit c
        ⊢ Eq
            (CategoryTheory.CategoryStruct.comp (CategoryTheory.section_ f)
              (let c' := CategoryTheory.Limits.KernelFork.ofι (CategoryTheory.Limits.F …
              let i' := CategoryTheory.Limits.isKernelCompMono i (CategoryTheory.secti …
              let i'' := CategoryTheory.Preadditive.isLimitForkOfKernelFork i';
              (CategoryTheory.Limits.splitMonoOfIdempotentOfIsLimitFork C ⋯ i'').retra …
            0
      -/
      dsimp only
      rw [splitMonoOfIdempotentOfIsLimitFork_retraction, isLimitForkOfKernelFork_lift,
        isKernelCompMono_lift]
      /-
        C : Type u
        inst✝³ : CategoryTheory.Category.{v, u} C
        inst✝² : CategoryTheory.Preadditive C
        X✝ Y✝ : C
        inst✝¹ : CategoryTheory.Limits.HasBinaryBiproduct X✝ Y✝
        X Y : C
        f : Quiver.Hom X Y
        inst✝ : CategoryTheory.IsSplitEpi f
        c : CategoryTheory.Limits.KernelFork f
        i : CategoryTheory.Limits.IsLimit c
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.section_ f) (i.lift ( …
      -/
      dsimp only [kernelForkOfFork_ι]
      /-
        C : Type u
        inst✝³ : CategoryTheory.Category.{v, u} C
        inst✝² : CategoryTheory.Preadditive C
        X✝ Y✝ : C
        inst✝¹ : CategoryTheory.Limits.HasBinaryBiproduct X✝ Y✝
        X Y : C
        f : Quiver.Hom X Y
        inst✝ : CategoryTheory.IsSplitEpi f
        c : CategoryTheory.Limits.KernelFork f
        i : CategoryTheory.Limits.IsLimit c
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.section_ f) (i.lift ( …
      -/
      letI := mono_of_isLimit_fork i
      /-
        C : Type u
        inst✝³ : CategoryTheory.Category.{v, u} C
        inst✝² : CategoryTheory.Preadditive C
        X✝ Y✝ : C
        inst✝¹ : CategoryTheory.Limits.HasBinaryBiproduct X✝ Y✝
        X Y : C
        f : Quiver.Hom X Y
        inst✝ : CategoryTheory.IsSplitEpi f
        c : CategoryTheory.Limits.KernelFork f
        i : CategoryTheory.Limits.IsLimit c
        this : CategoryTheory.Mono (CategoryTheory.Limits.Fork.ι c) := CategoryTheory. …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.section_ f) (i.lift ( …
      -/
      apply zero_of_comp_mono c.ι
      simp only [comp_sub, Category.comp_id, Category.assoc, sub_self, Fork.IsLimit.lift_ι,
        Fork.ι_ofι, IsSplitEpi.id_assoc]
                  /-
                    C : Type u
                    inst✝³ : CategoryTheory.Category.{v, u} C
                    inst✝² : CategoryTheory.Preadditive C
                    X✝ Y✝ : C
                    inst✝¹ : CategoryTheory.Limits.HasBinaryBiproduct X✝ Y✝
                    X Y : C
                    f : Quiver.Hom X Y
                    inst✝ : CategoryTheory.IsSplitEpi f
                    c : CategoryTheory.Limits.KernelFork f
                    i : CategoryTheory.Limits.IsLimit c
                    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.section_ f) f) (Categ …
                  -/
    inr_snd := by simp }
                  /-
                    🎉 no goals
                  -/


/-- The bicone constructed in `binaryBiconeOfIsSplitEpiOfKernel` is a bilimit.
This is a version of the splitting lemma that holds in all preadditive categories. -/
def isBilimitBinaryBiconeOfIsSplitEpiOfKernel {X Y : C} {f : X ⟶ Y} [IsSplitEpi f]
    {c : KernelFork f} (i : IsLimit c) : (binaryBiconeOfIsSplitEpiOfKernel i).IsBilimit :=
                                                                                   /-
                                                                                     C : Type u
                                                                                     inst✝³ : CategoryTheory.Category.{v, u} C
                                                                                     inst✝² : CategoryTheory.Preadditive C
                                                                                     X✝ Y✝ : C
                                                                                     inst✝¹ : CategoryTheory.Limits.HasBinaryBiproduct X✝ Y✝
                                                                                     X Y : C
                                                                                     f : Quiver.Hom X Y
                                                                                     inst✝ : CategoryTheory.IsSplitEpi f
                                                                                     c : CategoryTheory.Limits.KernelFork f
                                                                                     i : CategoryTheory.Limits.IsLimit c
                                                                                     ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Iso.refl c.pt).hom (C …
                                                                                   -/
  BinaryBicone.isBilimitOfKernelInl _ <| i.ofIsoLimit <| Fork.ext (Iso.refl _) (by simp)
                                                                                   /-
                                                                                     🎉 no goals
                                                                                   -/


/-- The existence of binary biproducts implies that there is at most one preadditive structure. -/
theorem biprod.add_eq_lift_id_desc [HasBinaryBiproduct X X] :
                                                            /-
                                                              C : Type u
                                                              inst✝² : CategoryTheory.Category.{v, u} C
                                                              inst✝¹ : CategoryTheory.Preadditive C
                                                              X Y : C
                                                              f g : Quiver.Hom X Y
                                                              inst✝ : CategoryTheory.Limits.HasBinaryBiproduct X X
                                                              ⊢ Eq (HAdd.hAdd f g) (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limit …
                                                            -/
    f + g = biprod.lift (𝟙 X) (𝟙 X) ≫ biprod.desc f g := by simp
                                                            /-
                                                              🎉 no goals
                                                            -/


/-- The existence of binary biproducts implies that there is at most one preadditive structure. -/
theorem biprod.add_eq_lift_desc_id [HasBinaryBiproduct Y Y] :
                                                            /-
                                                              C : Type u
                                                              inst✝² : CategoryTheory.Category.{v, u} C
                                                              inst✝¹ : CategoryTheory.Preadditive C
                                                              X Y : C
                                                              f g : Quiver.Hom X Y
                                                              inst✝ : CategoryTheory.Limits.HasBinaryBiproduct Y Y
                                                              ⊢ Eq (HAdd.hAdd f g) (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limit …
                                                            -/
    f + g = biprod.lift f g ≫ biprod.desc (𝟙 Y) (𝟙 Y) := by simp
                                                            /-
                                                              🎉 no goals
                                                            -/


attribute [local ext] Preadditive


/-- The existence of binary biproducts implies that there is at most one preadditive structure. -/
instance subsingleton_preadditive_of_hasBinaryBiproducts {C : Type u} [Category.{v} C]
    [HasZeroMorphisms C] [HasBinaryBiproducts C] : Subsingleton (Preadditive C) where
  allEq := fun a b => by
    /-
      C✝ : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C✝
      inst✝³ : CategoryTheory.Preadditive C✝
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝ : CategoryTheory.Limits.HasBinaryBiproducts C
      a b : CategoryTheory.Preadditive C
      ⊢ Eq a b
    -/
    apply Preadditive.ext; funext X Y; apply AddCommGroup.ext; funext f g
    have h₁ := @biprod.add_eq_lift_id_desc _ _ a _ _ f g
      (by convert (inferInstance : HasBinaryBiproduct X X); subsingleton)
    have h₂ := @biprod.add_eq_lift_id_desc _ _ b _ _ f g
      (by convert (inferInstance : HasBinaryBiproduct X X); subsingleton)
    /-
      case homGroup.h.h.h_mul.h.h
      C✝ : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C✝
      inst✝³ : CategoryTheory.Preadditive C✝
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝ : CategoryTheory.Limits.HasBinaryBiproducts C
      a b : CategoryTheory.Preadditive C
      X Y : C
      f g : Quiver.Hom X Y
      h₁ : Eq (HAdd.hAdd f g) (CategoryTheory.CategoryStruct.comp (CategoryTheory.Li …
      h₂ : Eq (HAdd.hAdd f g) (CategoryTheory.CategoryStruct.comp (CategoryTheory.Li …
      ⊢ Eq (Add.add f g) (Add.add f g)
    -/
    refine h₁.trans (Eq.trans ?_ h₂.symm)
    /-
      case homGroup.h.h.h_mul.h.h
      C✝ : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C✝
      inst✝³ : CategoryTheory.Preadditive C✝
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝ : CategoryTheory.Limits.HasBinaryBiproducts C
      a b : CategoryTheory.Preadditive C
      X Y : C
      f g : Quiver.Hom X Y
      h₁ : Eq (HAdd.hAdd f g) (CategoryTheory.CategoryStruct.comp (CategoryTheory.Li …
      h₂ : Eq (HAdd.hAdd f g) (CategoryTheory.CategoryStruct.comp (CategoryTheory.Li …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.biprod.lift (C …
    -/
                 /-
                   🎉 no goals
                 -/
                 /-
                   🎉 no goals
                 -/
    congr! 2 <;> subsingleton
                 /-
                   🎉 no goals
                 -/


/-- The "matrix" morphism `X₁ ⊞ X₂ ⟶ Y₁ ⊞ Y₂` with specified components.
-/
def Biprod.ofComponents : X₁ ⊞ X₂ ⟶ Y₁ ⊞ Y₂ :=
  biprod.fst ≫ f₁₁ ≫ biprod.inl + biprod.fst ≫ f₁₂ ≫ biprod.inr + biprod.snd ≫ f₂₁ ≫ biprod.inl +
    biprod.snd ≫ f₂₂ ≫ biprod.inr


@[simp]
theorem Biprod.inl_ofComponents :
    biprod.inl ≫ Biprod.ofComponents f₁₁ f₁₂ f₂₁ f₂₂ = f₁₁ ≫ biprod.inl + f₁₂ ≫ biprod.inr := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : CategoryTheory.Limits.HasBinaryBiproducts C
    X₁ X₂ Y₁ Y₂ : C
    f₁₁ : Quiver.Hom X₁ Y₁
    f₁₂ : Quiver.Hom X₁ Y₂
    f₂₁ : Quiver.Hom X₂ Y₁
    f₂₂ : Quiver.Hom X₂ Y₂
    ⊢ Eq (CategoryTheory.CategoryStruct.comp CategoryTheory.Limits.biprod.inl (Cat …
  -/
  simp [Biprod.ofComponents]
  /-
    🎉 no goals
  -/


@[simp]
theorem Biprod.inr_ofComponents :
    biprod.inr ≫ Biprod.ofComponents f₁₁ f₁₂ f₂₁ f₂₂ = f₂₁ ≫ biprod.inl + f₂₂ ≫ biprod.inr := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : CategoryTheory.Limits.HasBinaryBiproducts C
    X₁ X₂ Y₁ Y₂ : C
    f₁₁ : Quiver.Hom X₁ Y₁
    f₁₂ : Quiver.Hom X₁ Y₂
    f₂₁ : Quiver.Hom X₂ Y₁
    f₂₂ : Quiver.Hom X₂ Y₂
    ⊢ Eq (CategoryTheory.CategoryStruct.comp CategoryTheory.Limits.biprod.inr (Cat …
  -/
  simp [Biprod.ofComponents]
  /-
    🎉 no goals
  -/


@[simp]
theorem Biprod.ofComponents_fst :
    Biprod.ofComponents f₁₁ f₁₂ f₂₁ f₂₂ ≫ biprod.fst = biprod.fst ≫ f₁₁ + biprod.snd ≫ f₂₁ := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : CategoryTheory.Limits.HasBinaryBiproducts C
    X₁ X₂ Y₁ Y₂ : C
    f₁₁ : Quiver.Hom X₁ Y₁
    f₁₂ : Quiver.Hom X₁ Y₂
    f₂₁ : Quiver.Hom X₂ Y₁
    f₂₂ : Quiver.Hom X₂ Y₂
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Biprod.ofComponents f …
  -/
  simp [Biprod.ofComponents]
  /-
    🎉 no goals
  -/


@[simp]
theorem Biprod.ofComponents_snd :
    Biprod.ofComponents f₁₁ f₁₂ f₂₁ f₂₂ ≫ biprod.snd = biprod.fst ≫ f₁₂ + biprod.snd ≫ f₂₂ := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : CategoryTheory.Limits.HasBinaryBiproducts C
    X₁ X₂ Y₁ Y₂ : C
    f₁₁ : Quiver.Hom X₁ Y₁
    f₁₂ : Quiver.Hom X₁ Y₂
    f₂₁ : Quiver.Hom X₂ Y₁
    f₂₂ : Quiver.Hom X₂ Y₂
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Biprod.ofComponents f …
  -/
  simp [Biprod.ofComponents]
  /-
    🎉 no goals
  -/


@[simp]
theorem Biprod.ofComponents_eq (f : X₁ ⊞ X₂ ⟶ Y₁ ⊞ Y₂) :
    Biprod.ofComponents (biprod.inl ≫ f ≫ biprod.fst) (biprod.inl ≫ f ≫ biprod.snd)
        (biprod.inr ≫ f ≫ biprod.fst) (biprod.inr ≫ f ≫ biprod.snd) =
      f := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : CategoryTheory.Limits.HasBinaryBiproducts C
    X₁ X₂ Y₁ Y₂ : C
    f : Quiver.Hom (CategoryTheory.Limits.biprod X₁ X₂) (CategoryTheory.Limits.bip …
    ⊢ Eq (CategoryTheory.Biprod.ofComponents (CategoryTheory.CategoryStruct.comp C …
  -/
  ext <;>
    simp only [Category.comp_id, biprod.inr_fst, biprod.inr_snd, biprod.inl_snd, add_zero, zero_add,
      Biprod.inl_ofComponents, Biprod.inr_ofComponents, eq_self_iff_true, Category.assoc,
      comp_zero, biprod.inl_fst, Preadditive.add_comp]


@[simp]
theorem Biprod.ofComponents_comp {X₁ X₂ Y₁ Y₂ Z₁ Z₂ : C} (f₁₁ : X₁ ⟶ Y₁) (f₁₂ : X₁ ⟶ Y₂)
    (f₂₁ : X₂ ⟶ Y₁) (f₂₂ : X₂ ⟶ Y₂) (g₁₁ : Y₁ ⟶ Z₁) (g₁₂ : Y₁ ⟶ Z₂) (g₂₁ : Y₂ ⟶ Z₁)
    (g₂₂ : Y₂ ⟶ Z₂) :
    Biprod.ofComponents f₁₁ f₁₂ f₂₁ f₂₂ ≫ Biprod.ofComponents g₁₁ g₁₂ g₂₁ g₂₂ =
      Biprod.ofComponents (f₁₁ ≫ g₁₁ + f₁₂ ≫ g₂₁) (f₁₁ ≫ g₁₂ + f₁₂ ≫ g₂₂) (f₂₁ ≫ g₁₁ + f₂₂ ≫ g₂₁)
        (f₂₁ ≫ g₁₂ + f₂₂ ≫ g₂₂) := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : CategoryTheory.Limits.HasBinaryBiproducts C
    X₁ X₂ Y₁ Y₂ Z₁ Z₂ : C
    f₁₁ : Quiver.Hom X₁ Y₁
    f₁₂ : Quiver.Hom X₁ Y₂
    f₂₁ : Quiver.Hom X₂ Y₁
    f₂₂ : Quiver.Hom X₂ Y₂
    g₁₁ : Quiver.Hom Y₁ Z₁
    g₁₂ : Quiver.Hom Y₁ Z₂
    g₂₁ : Quiver.Hom Y₂ Z₁
    g₂₂ : Quiver.Hom Y₂ Z₂
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Biprod.ofComponents f …
  -/
  dsimp [Biprod.ofComponents]
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : CategoryTheory.Limits.HasBinaryBiproducts C
    X₁ X₂ Y₁ Y₂ Z₁ Z₂ : C
    f₁₁ : Quiver.Hom X₁ Y₁
    f₁₂ : Quiver.Hom X₁ Y₂
    f₂₁ : Quiver.Hom X₂ Y₁
    f₂₂ : Quiver.Hom X₂ Y₂
    g₁₁ : Quiver.Hom Y₁ Z₁
    g₁₂ : Quiver.Hom Y₁ Z₂
    g₂₁ : Quiver.Hom Y₂ Z₁
    g₂₂ : Quiver.Hom Y₂ Z₂
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (Cat …
  -/
  ext <;>
    simp only [add_comp, comp_add, add_comp_assoc, add_zero, zero_add, biprod.inl_fst,
      biprod.inl_snd, biprod.inr_fst, biprod.inr_snd, biprod.inl_fst_assoc, biprod.inl_snd_assoc,
      biprod.inr_fst_assoc, biprod.inr_snd_assoc, comp_zero, zero_comp, Category.assoc]


/-- The unipotent upper triangular matrix
```
(1 r)
(0 1)
```
as an isomorphism.
-/
@[simps]
def Biprod.unipotentUpper {X₁ X₂ : C} (r : X₁ ⟶ X₂) : X₁ ⊞ X₂ ≅ X₁ ⊞ X₂ where
  hom := Biprod.ofComponents (𝟙 _) r 0 (𝟙 _)
  inv := Biprod.ofComponents (𝟙 _) (-r) 0 (𝟙 _)


/-- The unipotent lower triangular matrix
```
(1 0)
(r 1)
```
as an isomorphism.
-/
@[simps]
def Biprod.unipotentLower {X₁ X₂ : C} (r : X₂ ⟶ X₁) : X₁ ⊞ X₂ ≅ X₁ ⊞ X₂ where
  hom := Biprod.ofComponents (𝟙 _) 0 r (𝟙 _)
  inv := Biprod.ofComponents (𝟙 _) 0 (-r) (𝟙 _)


/-- If `f` is a morphism `X₁ ⊞ X₂ ⟶ Y₁ ⊞ Y₂` whose `X₁ ⟶ Y₁` entry is an isomorphism,
then we can construct isomorphisms `L : X₁ ⊞ X₂ ≅ X₁ ⊞ X₂` and `R : Y₁ ⊞ Y₂ ≅ Y₁ ⊞ Y₂`
so that `L.hom ≫ g ≫ R.hom` is diagonal (with `X₁ ⟶ Y₁` component still `f`),
via Gaussian elimination.

(This is the version of `Biprod.gaussian` written in terms of components.)
-/
def Biprod.gaussian' [IsIso f₁₁] :
    Σ' (L : X₁ ⊞ X₂ ≅ X₁ ⊞ X₂) (R : Y₁ ⊞ Y₂ ≅ Y₁ ⊞ Y₂) (g₂₂ : X₂ ⟶ Y₂),
      L.hom ≫ Biprod.ofComponents f₁₁ f₁₂ f₂₁ f₂₂ ≫ R.hom = biprod.map f₁₁ g₂₂ :=
  ⟨Biprod.unipotentLower (-f₂₁ ≫ inv f₁₁), Biprod.unipotentUpper (-inv f₁₁ ≫ f₁₂),
                                  /-
                                    C : Type u
                                    inst✝³ : CategoryTheory.Category.{v, u} C
                                    inst✝² : CategoryTheory.Preadditive C
                                    inst✝¹ : CategoryTheory.Limits.HasBinaryBiproducts C
                                    X₁ X₂ Y₁ Y₂ : C
                                    f₁₁ : Quiver.Hom X₁ Y₁
                                    f₁₂ : Quiver.Hom X₁ Y₂
                                    f₂₁ : Quiver.Hom X₂ Y₁
                                    f₂₂ : Quiver.Hom X₂ Y₂
                                    inst✝ : CategoryTheory.IsIso f₁₁
                                    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Biprod.unipotentLower …
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
                                                /-
                                                  🎉 no goals
                                                -/
    f₂₂ - f₂₁ ≫ inv f₁₁ ≫ f₁₂, by ext <;> simp; abel⟩
                                                /-
                                                  🎉 no goals
                                                -/


/-- If `f` is a morphism `X₁ ⊞ X₂ ⟶ Y₁ ⊞ Y₂` whose `X₁ ⟶ Y₁` entry is an isomorphism,
then we can construct isomorphisms `L : X₁ ⊞ X₂ ≅ X₁ ⊞ X₂` and `R : Y₁ ⊞ Y₂ ≅ Y₁ ⊞ Y₂`
so that `L.hom ≫ g ≫ R.hom` is diagonal (with `X₁ ⟶ Y₁` component still `f`),
via Gaussian elimination.
-/
def Biprod.gaussian (f : X₁ ⊞ X₂ ⟶ Y₁ ⊞ Y₂) [IsIso (biprod.inl ≫ f ≫ biprod.fst)] :
    Σ' (L : X₁ ⊞ X₂ ≅ X₁ ⊞ X₂) (R : Y₁ ⊞ Y₂ ≅ Y₁ ⊞ Y₂) (g₂₂ : X₂ ⟶ Y₂),
      L.hom ≫ f ≫ R.hom = biprod.map (biprod.inl ≫ f ≫ biprod.fst) g₂₂ := by
  let this :=
    Biprod.gaussian' (biprod.inl ≫ f ≫ biprod.fst) (biprod.inl ≫ f ≫ biprod.snd)
      (biprod.inr ≫ f ≫ biprod.fst) (biprod.inr ≫ f ≫ biprod.snd)
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Preadditive C
    inst✝¹ : CategoryTheory.Limits.HasBinaryBiproducts C
    X₁ X₂ Y₁ Y₂ : C
    f₁₁ : Quiver.Hom X₁ Y₁
    f₁₂ : Quiver.Hom X₁ Y₂
    f₂₁ : Quiver.Hom X₂ Y₁
    f₂₂ : Quiver.Hom X₂ Y₂
    f : Quiver.Hom (CategoryTheory.Limits.biprod X₁ X₂) (CategoryTheory.Limits.bip …
    inst✝ : CategoryTheory.IsIso (CategoryTheory.CategoryStruct.comp CategoryTheor …
    this : PSigma fun L => PSigma fun R => PSigma fun g₂₂ => Eq (CategoryTheory.Ca …
    ⊢ PSigma fun L => PSigma fun R => PSigma fun g₂₂ => Eq (CategoryTheory.Categor …
  -/
  rwa [Biprod.ofComponents_eq] at this
  /-
    🎉 no goals
  -/


/-- If `X₁ ⊞ X₂ ≅ Y₁ ⊞ Y₂` via a two-by-two matrix whose `X₁ ⟶ Y₁` entry is an isomorphism,
then we can construct an isomorphism `X₂ ≅ Y₂`, via Gaussian elimination.
-/
def Biprod.isoElim' [IsIso f₁₁] [IsIso (Biprod.ofComponents f₁₁ f₁₂ f₂₁ f₂₂)] : X₂ ≅ Y₂ := by
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.Preadditive C
    inst✝² : CategoryTheory.Limits.HasBinaryBiproducts C
    X₁ X₂ Y₁ Y₂ : C
    f₁₁ : Quiver.Hom X₁ Y₁
    f₁₂ : Quiver.Hom X₁ Y₂
    f₂₁ : Quiver.Hom X₂ Y₁
    f₂₂ : Quiver.Hom X₂ Y₂
    inst✝¹ : CategoryTheory.IsIso f₁₁
    inst✝ : CategoryTheory.IsIso (CategoryTheory.Biprod.ofComponents f₁₁ f₁₂ f₂₁ f …
    ⊢ CategoryTheory.Iso X₂ Y₂
  -/
  obtain ⟨L, R, g, w⟩ := Biprod.gaussian' f₁₁ f₁₂ f₂₁ f₂₂
  letI : IsIso (biprod.map f₁₁ g) := by
    rw [← w]
    infer_instance
  /-
    case mk.mk.mk
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.Preadditive C
    inst✝² : CategoryTheory.Limits.HasBinaryBiproducts C
    X₁ X₂ Y₁ Y₂ : C
    f₁₁ : Quiver.Hom X₁ Y₁
    f₁₂ : Quiver.Hom X₁ Y₂
    f₂₁ : Quiver.Hom X₂ Y₁
    f₂₂ : Quiver.Hom X₂ Y₂
    inst✝¹ : CategoryTheory.IsIso f₁₁
    inst✝ : CategoryTheory.IsIso (CategoryTheory.Biprod.ofComponents f₁₁ f₁₂ f₂₁ f …
    L : CategoryTheory.Iso (CategoryTheory.Limits.biprod X₁ X₂) (CategoryTheory.Li …
    R : CategoryTheory.Iso (CategoryTheory.Limits.biprod Y₁ Y₂) (CategoryTheory.Li …
    g : Quiver.Hom X₂ Y₂
    w : Eq (CategoryTheory.CategoryStruct.comp L.hom (CategoryTheory.CategoryStruc …
    this : CategoryTheory.IsIso (CategoryTheory.Limits.biprod.map f₁₁ g) := Eq.mpr …
    ⊢ CategoryTheory.Iso X₂ Y₂
  -/
  letI : IsIso g := isIso_right_of_isIso_biprod_map f₁₁ g
  /-
    case mk.mk.mk
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.Preadditive C
    inst✝² : CategoryTheory.Limits.HasBinaryBiproducts C
    X₁ X₂ Y₁ Y₂ : C
    f₁₁ : Quiver.Hom X₁ Y₁
    f₁₂ : Quiver.Hom X₁ Y₂
    f₂₁ : Quiver.Hom X₂ Y₁
    f₂₂ : Quiver.Hom X₂ Y₂
    inst✝¹ : CategoryTheory.IsIso f₁₁
    inst✝ : CategoryTheory.IsIso (CategoryTheory.Biprod.ofComponents f₁₁ f₁₂ f₂₁ f …
    L : CategoryTheory.Iso (CategoryTheory.Limits.biprod X₁ X₂) (CategoryTheory.Li …
    R : CategoryTheory.Iso (CategoryTheory.Limits.biprod Y₁ Y₂) (CategoryTheory.Li …
    g : Quiver.Hom X₂ Y₂
    w : Eq (CategoryTheory.CategoryStruct.comp L.hom (CategoryTheory.CategoryStruc …
    this✝ : CategoryTheory.IsIso (CategoryTheory.Limits.biprod.map f₁₁ g) := Eq.mp …
    this : CategoryTheory.IsIso g := CategoryTheory.isIso_right_of_isIso_biprod_ma …
    ⊢ CategoryTheory.Iso X₂ Y₂
  -/
  exact asIso g
  /-
    🎉 no goals
  -/


/-- If `f` is an isomorphism `X₁ ⊞ X₂ ≅ Y₁ ⊞ Y₂` whose `X₁ ⟶ Y₁` entry is an isomorphism,
then we can construct an isomorphism `X₂ ≅ Y₂`, via Gaussian elimination.
-/
def Biprod.isoElim (f : X₁ ⊞ X₂ ≅ Y₁ ⊞ Y₂) [IsIso (biprod.inl ≫ f.hom ≫ biprod.fst)] : X₂ ≅ Y₂ :=
  letI :
    IsIso
      (Biprod.ofComponents (biprod.inl ≫ f.hom ≫ biprod.fst) (biprod.inl ≫ f.hom ≫ biprod.snd)
        (biprod.inr ≫ f.hom ≫ biprod.fst) (biprod.inr ≫ f.hom ≫ biprod.snd)) := by
    /-
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.Preadditive C
      inst✝¹ : CategoryTheory.Limits.HasBinaryBiproducts C
      X₁ X₂ Y₁ Y₂ : C
      f₁₁ : Quiver.Hom X₁ Y₁
      f₁₂ : Quiver.Hom X₁ Y₂
      f₂₁ : Quiver.Hom X₂ Y₁
      f₂₂ : Quiver.Hom X₂ Y₂
      f : CategoryTheory.Iso (CategoryTheory.Limits.biprod X₁ X₂) (CategoryTheory.Li …
      inst✝ : CategoryTheory.IsIso (CategoryTheory.CategoryStruct.comp CategoryTheor …
      ⊢ CategoryTheory.IsIso (CategoryTheory.Biprod.ofComponents (CategoryTheory.Cat …
    -/
    simp only [Biprod.ofComponents_eq]
    /-
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.Preadditive C
      inst✝¹ : CategoryTheory.Limits.HasBinaryBiproducts C
      X₁ X₂ Y₁ Y₂ : C
      f₁₁ : Quiver.Hom X₁ Y₁
      f₁₂ : Quiver.Hom X₁ Y₂
      f₂₁ : Quiver.Hom X₂ Y₁
      f₂₂ : Quiver.Hom X₂ Y₂
      f : CategoryTheory.Iso (CategoryTheory.Limits.biprod X₁ X₂) (CategoryTheory.Li …
      inst✝ : CategoryTheory.IsIso (CategoryTheory.CategoryStruct.comp CategoryTheor …
      ⊢ CategoryTheory.IsIso f.hom
    -/
    infer_instance
    /-
      🎉 no goals
    -/
  Biprod.isoElim' (biprod.inl ≫ f.hom ≫ biprod.fst) (biprod.inl ≫ f.hom ≫ biprod.snd)
    (biprod.inr ≫ f.hom ≫ biprod.fst) (biprod.inr ≫ f.hom ≫ biprod.snd)


theorem Biprod.column_nonzero_of_iso {W X Y Z : C} (f : W ⊞ X ⟶ Y ⊞ Z) [IsIso f] :
    𝟙 W = 0 ∨ biprod.inl ≫ f ≫ biprod.fst ≠ 0 ∨ biprod.inl ≫ f ≫ biprod.snd ≠ 0 := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Preadditive C
    inst✝¹ : CategoryTheory.Limits.HasBinaryBiproducts C
    W X Y Z : C
    f : Quiver.Hom (CategoryTheory.Limits.biprod W X) (CategoryTheory.Limits.bipro …
    inst✝ : CategoryTheory.IsIso f
    ⊢ Or (Eq (CategoryTheory.CategoryStruct.id W) 0) (Or (Ne (CategoryTheory.Categ …
  -/
  by_contra! h
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Preadditive C
    inst✝¹ : CategoryTheory.Limits.HasBinaryBiproducts C
    W X Y Z : C
    f : Quiver.Hom (CategoryTheory.Limits.biprod W X) (CategoryTheory.Limits.bipro …
    inst✝ : CategoryTheory.IsIso f
    h : And (Ne (CategoryTheory.CategoryStruct.id W) 0) (And (Eq (CategoryTheory.C …
    ⊢ False
  -/
  rcases h with ⟨nz, a₁, a₂⟩
  /-
    case intro.intro
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Preadditive C
    inst✝¹ : CategoryTheory.Limits.HasBinaryBiproducts C
    W X Y Z : C
    f : Quiver.Hom (CategoryTheory.Limits.biprod W X) (CategoryTheory.Limits.bipro …
    inst✝ : CategoryTheory.IsIso f
    nz : Ne (CategoryTheory.CategoryStruct.id W) 0
    a₁ : Eq (CategoryTheory.CategoryStruct.comp CategoryTheory.Limits.biprod.inl ( …
    a₂ : Eq (CategoryTheory.CategoryStruct.comp CategoryTheory.Limits.biprod.inl ( …
    ⊢ False
  -/
  set x := biprod.inl ≫ f ≫ inv f ≫ biprod.fst
  /-
    case intro.intro
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Preadditive C
    inst✝¹ : CategoryTheory.Limits.HasBinaryBiproducts C
    W X Y Z : C
    f : Quiver.Hom (CategoryTheory.Limits.biprod W X) (CategoryTheory.Limits.bipro …
    inst✝ : CategoryTheory.IsIso f
    nz : Ne (CategoryTheory.CategoryStruct.id W) 0
    a₁ : Eq (CategoryTheory.CategoryStruct.comp CategoryTheory.Limits.biprod.inl ( …
    a₂ : Eq (CategoryTheory.CategoryStruct.comp CategoryTheory.Limits.biprod.inl ( …
    x : Quiver.Hom W W := CategoryTheory.CategoryStruct.comp CategoryTheory.Limits …
    ⊢ False
  -/
  have h₁ : x = 𝟙 W := by simp [x]
  have h₀ : x = 0 := by
    dsimp [x]
    rw [← Category.id_comp (inv f), Category.assoc, ← biprod.total]
    conv_lhs =>
      slice 2 3
      rw [comp_add]
    simp only [Category.assoc]
    rw [comp_add_assoc, add_comp]
    conv_lhs =>
      congr
      next => skip
      slice 1 3
      rw [a₂]
    simp only [zero_comp, add_zero]
    conv_lhs =>
      slice 1 3
      rw [a₁]
    simp only [zero_comp]
  /-
    case intro.intro
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Preadditive C
    inst✝¹ : CategoryTheory.Limits.HasBinaryBiproducts C
    W X Y Z : C
    f : Quiver.Hom (CategoryTheory.Limits.biprod W X) (CategoryTheory.Limits.bipro …
    inst✝ : CategoryTheory.IsIso f
    nz : Ne (CategoryTheory.CategoryStruct.id W) 0
    a₁ : Eq (CategoryTheory.CategoryStruct.comp CategoryTheory.Limits.biprod.inl ( …
    a₂ : Eq (CategoryTheory.CategoryStruct.comp CategoryTheory.Limits.biprod.inl ( …
    x : Quiver.Hom W W := CategoryTheory.CategoryStruct.comp CategoryTheory.Limits …
    h₁ : Eq x (CategoryTheory.CategoryStruct.id W)
    h₀ : Eq x 0
    ⊢ False
  -/
  exact nz (h₁.symm.trans h₀)
  /-
    🎉 no goals
  -/


theorem Biproduct.column_nonzero_of_iso' {σ τ : Type} [Finite τ] {S : σ → C} [HasBiproduct S]
    {T : τ → C} [HasBiproduct T] (s : σ) (f : ⨁ S ⟶ ⨁ T) [IsIso f] :
    (∀ t : τ, biproduct.ι S s ≫ f ≫ biproduct.π T t = 0) → 𝟙 (S s) = 0 := by
  /-
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    inst✝⁴ : CategoryTheory.Preadditive C
    σ τ : Type
    inst✝³ : Finite τ
    S : σ → C
    inst✝² : CategoryTheory.Limits.HasBiproduct S
    T : τ → C
    inst✝¹ : CategoryTheory.Limits.HasBiproduct T
    s : σ
    f : Quiver.Hom (CategoryTheory.Limits.biproduct S) (CategoryTheory.Limits.bipr …
    inst✝ : CategoryTheory.IsIso f
    ⊢ (∀ (t : τ), Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.bi …
  -/
  cases nonempty_fintype τ
  /-
    case intro
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    inst✝⁴ : CategoryTheory.Preadditive C
    σ τ : Type
    inst✝³ : Finite τ
    S : σ → C
    inst✝² : CategoryTheory.Limits.HasBiproduct S
    T : τ → C
    inst✝¹ : CategoryTheory.Limits.HasBiproduct T
    s : σ
    f : Quiver.Hom (CategoryTheory.Limits.biproduct S) (CategoryTheory.Limits.bipr …
    inst✝ : CategoryTheory.IsIso f
    val✝ : Fintype τ
    ⊢ (∀ (t : τ), Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.bi …
  -/
  intro z
  have reassoced {t : τ} {W : C} (h : _ ⟶ W) :
    biproduct.ι S s ≫ f ≫ biproduct.π T t ≫ h = 0 ≫ h := by
    simp only [← Category.assoc]
    apply eq_whisker
    simp only [Category.assoc]
    apply z
  /-
    case intro
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    inst✝⁴ : CategoryTheory.Preadditive C
    σ τ : Type
    inst✝³ : Finite τ
    S : σ → C
    inst✝² : CategoryTheory.Limits.HasBiproduct S
    T : τ → C
    inst✝¹ : CategoryTheory.Limits.HasBiproduct T
    s : σ
    f : Quiver.Hom (CategoryTheory.Limits.biproduct S) (CategoryTheory.Limits.bipr …
    inst✝ : CategoryTheory.IsIso f
    val✝ : Fintype τ
    z : ∀ (t : τ), Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.b …
    reassoced : ∀ {t : τ} {W : C} (h : Quiver.Hom (T t) W), Eq (CategoryTheory.Cat …
    ⊢ Eq (CategoryTheory.CategoryStruct.id (S s)) 0
  -/
  set x := biproduct.ι S s ≫ f ≫ inv f ≫ biproduct.π S s
  /-
    case intro
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    inst✝⁴ : CategoryTheory.Preadditive C
    σ τ : Type
    inst✝³ : Finite τ
    S : σ → C
    inst✝² : CategoryTheory.Limits.HasBiproduct S
    T : τ → C
    inst✝¹ : CategoryTheory.Limits.HasBiproduct T
    s : σ
    f : Quiver.Hom (CategoryTheory.Limits.biproduct S) (CategoryTheory.Limits.bipr …
    inst✝ : CategoryTheory.IsIso f
    val✝ : Fintype τ
    z : ∀ (t : τ), Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.b …
    reassoced : ∀ {t : τ} {W : C} (h : Quiver.Hom (T t) W), Eq (CategoryTheory.Cat …
    x : Quiver.Hom (S s) (S s) := CategoryTheory.CategoryStruct.comp (CategoryTheo …
    ⊢ Eq (CategoryTheory.CategoryStruct.id (S s)) 0
  -/
  have h₁ : x = 𝟙 (S s) := by simp [x]
  have h₀ : x = 0 := by
    dsimp [x]
    rw [← Category.id_comp (inv f), Category.assoc, ← biproduct.total]
    simp only [comp_sum_assoc]
    conv_lhs =>
      congr
      congr
      next => skip
      intro j; simp only [reassoced]
    simp
  /-
    case intro
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    inst✝⁴ : CategoryTheory.Preadditive C
    σ τ : Type
    inst✝³ : Finite τ
    S : σ → C
    inst✝² : CategoryTheory.Limits.HasBiproduct S
    T : τ → C
    inst✝¹ : CategoryTheory.Limits.HasBiproduct T
    s : σ
    f : Quiver.Hom (CategoryTheory.Limits.biproduct S) (CategoryTheory.Limits.bipr …
    inst✝ : CategoryTheory.IsIso f
    val✝ : Fintype τ
    z : ∀ (t : τ), Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.b …
    reassoced : ∀ {t : τ} {W : C} (h : Quiver.Hom (T t) W), Eq (CategoryTheory.Cat …
    x : Quiver.Hom (S s) (S s) := CategoryTheory.CategoryStruct.comp (CategoryTheo …
    h₁ : Eq x (CategoryTheory.CategoryStruct.id (S s))
    h₀ : Eq x 0
    ⊢ Eq (CategoryTheory.CategoryStruct.id (S s)) 0
  -/
  exact h₁.symm.trans h₀
  /-
    🎉 no goals
  -/


/-- If `f : ⨁ S ⟶ ⨁ T` is an isomorphism, and `s` is a non-trivial summand of the source,
then there is some `t` in the target so that the `s, t` matrix entry of `f` is nonzero.
-/
def Biproduct.columnNonzeroOfIso {σ τ : Type} [Fintype τ] {S : σ → C} [HasBiproduct S] {T : τ → C}
    [HasBiproduct T] (s : σ) (nz : 𝟙 (S s) ≠ 0) (f : ⨁ S ⟶ ⨁ T) [IsIso f] :
    Trunc (Σ't : τ, biproduct.ι S s ≫ f ≫ biproduct.π T t ≠ 0) := by
  classical
    apply truncSigmaOfExists
    have t := Biproduct.column_nonzero_of_iso'.{v} s f
    by_contra h
    simp only [not_exists_not] at h
    exact nz (t h)


/-- A functor between preadditive categories that preserves (zero morphisms and) finite biproducts
    preserves finite products. -/
lemma preservesProduct_of_preservesBiproduct {f : J → C} [PreservesBiproduct f F] :
    PreservesLimit (Discrete.functor f) F where
  preserves hc :=
    ⟨IsLimit.ofIsoLimit
        ((IsLimit.postcomposeInvEquiv (Discrete.compNatIsoDiscrete _ _) _).symm
          (isBilimitOfPreserves F (biconeIsBilimitOfLimitConeOfIsLimit hc)).isLimit) <|
                                 /-
                                   C : Type u
                                   inst✝⁶ : CategoryTheory.Category.{v, u} C
                                   inst✝⁵ : CategoryTheory.Preadditive C
                                   D : Type u'
                                   inst✝⁴ : CategoryTheory.Category.{v', u'} D
                                   inst✝³ : CategoryTheory.Preadditive D
                                   F : CategoryTheory.Functor C D
                                   inst✝² : F.PreservesZeroMorphisms
                                   J : Type
                                   inst✝¹ : Fintype J
                                   f : J → C
                                   inst✝ : CategoryTheory.Limits.PreservesBiproduct f F
                                   c✝ : CategoryTheory.Limits.Cone (CategoryTheory.Discrete.functor f)
                                   hc : CategoryTheory.Limits.IsLimit c✝
                                   ⊢ ∀ (j : CategoryTheory.Discrete J), Eq (((CategoryTheory.Limits.Cones.postcom …
                                 -/
      Cones.ext (Iso.refl _) (by rintro ⟨⟩; simp)⟩
                                            /-
                                              🎉 no goals
                                            -/


/-- A functor between preadditive categories that preserves (zero morphisms and) finite biproducts
    preserves finite products. -/
lemma preservesProductsOfShape_of_preservesBiproductsOfShape [PreservesBiproductsOfShape J F] :
    PreservesLimitsOfShape (Discrete J) F where
  preservesLimit {_} := preservesLimit_of_iso_diagram _ Discrete.natIsoFunctor.symm


/-- A functor between preadditive categories that preserves (zero morphisms and) finite products
    preserves finite biproducts. -/
lemma preservesBiproduct_of_preservesProduct {f : J → C} [PreservesLimit (Discrete.functor f) F] :
    PreservesBiproduct f F where
  preserves {b} hb :=
    ⟨isBilimitOfIsLimit _ <|
      IsLimit.ofIsoLimit
          ((IsLimit.postcomposeHomEquiv (Discrete.compNatIsoDiscrete _ _) (F.mapCone b.toCone)).symm
            (isLimitOfPreserves F hb.isLimit)) <|
                                   /-
                                     C : Type u
                                     inst✝⁶ : CategoryTheory.Category.{v, u} C
                                     inst✝⁵ : CategoryTheory.Preadditive C
                                     D : Type u'
                                     inst✝⁴ : CategoryTheory.Category.{v', u'} D
                                     inst✝³ : CategoryTheory.Preadditive D
                                     F : CategoryTheory.Functor C D
                                     inst✝² : F.PreservesZeroMorphisms
                                     J : Type
                                     inst✝¹ : Fintype J
                                     f : J → C
                                     inst✝ : CategoryTheory.Limits.PreservesLimit (CategoryTheory.Discrete.functor  …
                                     b : CategoryTheory.Limits.Bicone f
                                     hb : b.IsBilimit
                                     ⊢ ∀ (j : CategoryTheory.Discrete J), Eq (((CategoryTheory.Limits.Cones.postcom …
                                   -/
        Cones.ext (Iso.refl _) (by rintro ⟨⟩; simp)⟩
                                              /-
                                                🎉 no goals
                                              -/


/-- If the (product-like) biproduct comparison for `F` and `f` is a monomorphism, then `F`
    preserves the biproduct of `f`. For the converse, see `mapBiproduct`. -/
lemma preservesBiproduct_of_mono_biproductComparison {f : J → C} [HasBiproduct f]
    [HasBiproduct (F.obj ∘ f)] [Mono (biproductComparison F f)] : PreservesBiproduct f F := by
  haveI : HasProduct fun b => F.obj (f b) := by
    change HasProduct (F.obj ∘ f)
    infer_instance
  have that : piComparison F f =
      (F.mapIso (biproduct.isoProduct f)).inv ≫
        biproductComparison F f ≫ (biproduct.isoProduct _).hom := by
    ext j
    convert piComparison_comp_π F f j; simp [← Function.comp_def, ← Functor.map_comp]
  /-
    C : Type u
    inst✝⁸ : CategoryTheory.Category.{v, u} C
    inst✝⁷ : CategoryTheory.Preadditive C
    D : Type u'
    inst✝⁶ : CategoryTheory.Category.{v', u'} D
    inst✝⁵ : CategoryTheory.Preadditive D
    F : CategoryTheory.Functor C D
    inst✝⁴ : F.PreservesZeroMorphisms
    J : Type
    inst✝³ : Fintype J
    f : J → C
    inst✝² : CategoryTheory.Limits.HasBiproduct f
    inst✝¹ : CategoryTheory.Limits.HasBiproduct (Function.comp F.obj f)
    inst✝ : CategoryTheory.Mono (F.biproductComparison f)
    this : CategoryTheory.Limits.HasProduct fun b => F.obj (f b)
    that : Eq (CategoryTheory.Limits.piComparison F f) (CategoryTheory.CategoryStr …
    ⊢ CategoryTheory.Limits.PreservesBiproduct f F
  -/
  haveI : IsIso (biproductComparison F f) := isIso_of_mono_of_isSplitEpi _
  haveI : IsIso (piComparison F f) := by
    rw [that]
    infer_instance
  /-
    C : Type u
    inst✝⁸ : CategoryTheory.Category.{v, u} C
    inst✝⁷ : CategoryTheory.Preadditive C
    D : Type u'
    inst✝⁶ : CategoryTheory.Category.{v', u'} D
    inst✝⁵ : CategoryTheory.Preadditive D
    F : CategoryTheory.Functor C D
    inst✝⁴ : F.PreservesZeroMorphisms
    J : Type
    inst✝³ : Fintype J
    f : J → C
    inst✝² : CategoryTheory.Limits.HasBiproduct f
    inst✝¹ : CategoryTheory.Limits.HasBiproduct (Function.comp F.obj f)
    inst✝ : CategoryTheory.Mono (F.biproductComparison f)
    this✝¹ : CategoryTheory.Limits.HasProduct fun b => F.obj (f b)
    that : Eq (CategoryTheory.Limits.piComparison F f) (CategoryTheory.CategoryStr …
    this✝ : CategoryTheory.IsIso (F.biproductComparison f)
    this : CategoryTheory.IsIso (CategoryTheory.Limits.piComparison F f)
    ⊢ CategoryTheory.Limits.PreservesBiproduct f F
  -/
  haveI := PreservesProduct.of_iso_comparison F f
  /-
    C : Type u
    inst✝⁸ : CategoryTheory.Category.{v, u} C
    inst✝⁷ : CategoryTheory.Preadditive C
    D : Type u'
    inst✝⁶ : CategoryTheory.Category.{v', u'} D
    inst✝⁵ : CategoryTheory.Preadditive D
    F : CategoryTheory.Functor C D
    inst✝⁴ : F.PreservesZeroMorphisms
    J : Type
    inst✝³ : Fintype J
    f : J → C
    inst✝² : CategoryTheory.Limits.HasBiproduct f
    inst✝¹ : CategoryTheory.Limits.HasBiproduct (Function.comp F.obj f)
    inst✝ : CategoryTheory.Mono (F.biproductComparison f)
    this✝² : CategoryTheory.Limits.HasProduct fun b => F.obj (f b)
    that : Eq (CategoryTheory.Limits.piComparison F f) (CategoryTheory.CategoryStr …
    this✝¹ : CategoryTheory.IsIso (F.biproductComparison f)
    this✝ : CategoryTheory.IsIso (CategoryTheory.Limits.piComparison F f)
    this : CategoryTheory.Limits.PreservesLimit (CategoryTheory.Discrete.functor f …
    ⊢ CategoryTheory.Limits.PreservesBiproduct f F
  -/
  apply preservesBiproduct_of_preservesProduct
  /-
    🎉 no goals
  -/


/-- If the (coproduct-like) biproduct comparison for `F` and `f` is an epimorphism, then `F`
    preserves the biproduct of `F` and `f`. For the converse, see `mapBiproduct`. -/
lemma preservesBiproduct_of_epi_biproductComparison' {f : J → C} [HasBiproduct f]
    [HasBiproduct (F.obj ∘ f)] [Epi (biproductComparison' F f)] : PreservesBiproduct f F := by
  /-
    C : Type u
    inst✝⁸ : CategoryTheory.Category.{v, u} C
    inst✝⁷ : CategoryTheory.Preadditive C
    D : Type u'
    inst✝⁶ : CategoryTheory.Category.{v', u'} D
    inst✝⁵ : CategoryTheory.Preadditive D
    F : CategoryTheory.Functor C D
    inst✝⁴ : F.PreservesZeroMorphisms
    J : Type
    inst✝³ : Fintype J
    f : J → C
    inst✝² : CategoryTheory.Limits.HasBiproduct f
    inst✝¹ : CategoryTheory.Limits.HasBiproduct (Function.comp F.obj f)
    inst✝ : CategoryTheory.Epi (F.biproductComparison' f)
    ⊢ CategoryTheory.Limits.PreservesBiproduct f F
  -/
  haveI : Epi (splitEpiBiproductComparison F f).section_ := by simpa
  haveI : IsIso (biproductComparison F f) :=
    IsIso.of_epi_section' (splitEpiBiproductComparison F f)
  /-
    C : Type u
    inst✝⁸ : CategoryTheory.Category.{v, u} C
    inst✝⁷ : CategoryTheory.Preadditive C
    D : Type u'
    inst✝⁶ : CategoryTheory.Category.{v', u'} D
    inst✝⁵ : CategoryTheory.Preadditive D
    F : CategoryTheory.Functor C D
    inst✝⁴ : F.PreservesZeroMorphisms
    J : Type
    inst✝³ : Fintype J
    f : J → C
    inst✝² : CategoryTheory.Limits.HasBiproduct f
    inst✝¹ : CategoryTheory.Limits.HasBiproduct (Function.comp F.obj f)
    inst✝ : CategoryTheory.Epi (F.biproductComparison' f)
    this✝ : CategoryTheory.Epi (F.splitEpiBiproductComparison f).section_
    this : CategoryTheory.IsIso (F.biproductComparison f)
    ⊢ CategoryTheory.Limits.PreservesBiproduct f F
  -/
  apply preservesBiproduct_of_mono_biproductComparison
  /-
    🎉 no goals
  -/


/-- A functor between preadditive categories that preserves (zero morphisms and) finite products
    preserves finite biproducts. -/
lemma preservesBiproductsOfShape_of_preservesProductsOfShape
    [PreservesLimitsOfShape (Discrete J) F] :
    PreservesBiproductsOfShape J F where
  preserves {_} := preservesBiproduct_of_preservesProduct F


/-- A functor between preadditive categories that preserves (zero morphisms and) finite biproducts
    preserves finite coproducts. -/
lemma preservesCoproduct_of_preservesBiproduct {f : J → C} [PreservesBiproduct f F] :
    PreservesColimit (Discrete.functor f) F where
  preserves {c} hc :=
    ⟨IsColimit.ofIsoColimit
        ((IsColimit.precomposeHomEquiv (Discrete.compNatIsoDiscrete _ _) _).symm
          (isBilimitOfPreserves F (biconeIsBilimitOfColimitCoconeOfIsColimit hc)).isColimit) <|
                                   /-
                                     C : Type u
                                     inst✝⁶ : CategoryTheory.Category.{v, u} C
                                     inst✝⁵ : CategoryTheory.Preadditive C
                                     D : Type u'
                                     inst✝⁴ : CategoryTheory.Category.{v', u'} D
                                     inst✝³ : CategoryTheory.Preadditive D
                                     F : CategoryTheory.Functor C D
                                     inst✝² : F.PreservesZeroMorphisms
                                     J : Type
                                     inst✝¹ : Fintype J
                                     f : J → C
                                     inst✝ : CategoryTheory.Limits.PreservesBiproduct f F
                                     c : CategoryTheory.Limits.Cocone (CategoryTheory.Discrete.functor f)
                                     hc : CategoryTheory.Limits.IsColimit c
                                     ⊢ ∀ (j : CategoryTheory.Discrete J), Eq (CategoryTheory.CategoryStruct.comp (( …
                                   -/
      Cocones.ext (Iso.refl _) (by rintro ⟨⟩; simp)⟩
                                              /-
                                                🎉 no goals
                                              -/


/-- A functor between preadditive categories that preserves (zero morphisms and) finite biproducts
    preserves finite coproducts. -/
lemma preservesCoproductsOfShape_of_preservesBiproductsOfShape [PreservesBiproductsOfShape J F] :
    PreservesColimitsOfShape (Discrete J) F where
  preservesColimit {_} := preservesColimit_of_iso_diagram _ Discrete.natIsoFunctor.symm


/-- A functor between preadditive categories that preserves (zero morphisms and) finite coproducts
    preserves finite biproducts. -/
lemma preservesBiproduct_of_preservesCoproduct {f : J → C}
    [PreservesColimit (Discrete.functor f) F] :
    PreservesBiproduct f F where
  preserves {b} hb :=
    ⟨isBilimitOfIsColimit _ <|
      IsColimit.ofIsoColimit
          ((IsColimit.precomposeInvEquiv (Discrete.compNatIsoDiscrete _ _)
                (F.mapCocone b.toCocone)).symm
            (isColimitOfPreserves F hb.isColimit)) <|
                                     /-
                                       C : Type u
                                       inst✝⁶ : CategoryTheory.Category.{v, u} C
                                       inst✝⁵ : CategoryTheory.Preadditive C
                                       D : Type u'
                                       inst✝⁴ : CategoryTheory.Category.{v', u'} D
                                       inst✝³ : CategoryTheory.Preadditive D
                                       F : CategoryTheory.Functor C D
                                       inst✝² : F.PreservesZeroMorphisms
                                       J : Type
                                       inst✝¹ : Fintype J
                                       f : J → C
                                       inst✝ : CategoryTheory.Limits.PreservesColimit (CategoryTheory.Discrete.functo …
                                       b : CategoryTheory.Limits.Bicone f
                                       hb : b.IsBilimit
                                       ⊢ ∀ (j : CategoryTheory.Discrete J), Eq (CategoryTheory.CategoryStruct.comp (( …
                                     -/
        Cocones.ext (Iso.refl _) (by rintro ⟨⟩; simp)⟩
                                                /-
                                                  🎉 no goals
                                                -/


/-- A functor between preadditive categories that preserves (zero morphisms and) finite coproducts
    preserves finite biproducts. -/
lemma preservesBiproductsOfShape_of_preservesCoproductsOfShape
    [PreservesColimitsOfShape (Discrete J) F] : PreservesBiproductsOfShape J F where
  preserves {_} := preservesBiproduct_of_preservesCoproduct F


/-- A functor between preadditive categories that preserves (zero morphisms and) binary biproducts
    preserves binary products. -/
lemma preservesBinaryProduct_of_preservesBinaryBiproduct {X Y : C}
    [PreservesBinaryBiproduct X Y F] :
    PreservesLimit (pair X Y) F where
  preserves {c} hc := ⟨IsLimit.ofIsoLimit
        ((IsLimit.postcomposeInvEquiv (diagramIsoPair _) _).symm
          (isBinaryBilimitOfPreserves F (binaryBiconeIsBilimitOfLimitConeOfIsLimit hc)).isLimit) <|
                    /-
                      C : Type u
                      inst✝⁵ : CategoryTheory.Category.{v, u} C
                      inst✝⁴ : CategoryTheory.Preadditive C
                      D : Type u'
                      inst✝³ : CategoryTheory.Category.{v', u'} D
                      inst✝² : CategoryTheory.Preadditive D
                      F : CategoryTheory.Functor C D
                      inst✝¹ : F.PreservesZeroMorphisms
                      X Y : C
                      inst✝ : CategoryTheory.Limits.PreservesBinaryBiproduct X Y F
                      c : CategoryTheory.Limits.Cone (CategoryTheory.Limits.pair X Y)
                      hc : CategoryTheory.Limits.IsLimit c
                      ⊢ CategoryTheory.Iso ((CategoryTheory.Limits.Cones.postcompose (CategoryTheory …
                    -/
      Cones.ext (by dsimp; rfl) fun j => by
                           /-
                             🎉 no goals
                           -/
        /-
          C : Type u
          inst✝⁵ : CategoryTheory.Category.{v, u} C
          inst✝⁴ : CategoryTheory.Preadditive C
          D : Type u'
          inst✝³ : CategoryTheory.Category.{v', u'} D
          inst✝² : CategoryTheory.Preadditive D
          F : CategoryTheory.Functor C D
          inst✝¹ : F.PreservesZeroMorphisms
          X Y : C
          inst✝ : CategoryTheory.Limits.PreservesBinaryBiproduct X Y F
          c : CategoryTheory.Limits.Cone (CategoryTheory.Limits.pair X Y)
          hc : CategoryTheory.Limits.IsLimit c
          j : CategoryTheory.Discrete CategoryTheory.Limits.WalkingPair
          ⊢ Eq (((CategoryTheory.Limits.Cones.postcompose (CategoryTheory.Limits.diagram …
        -/
                               /-
                                 🎉 no goals
                               -/
        rcases j with ⟨⟨⟩⟩ <;> simp⟩
                               /-
                                 🎉 no goals
                               -/


/-- A functor between preadditive categories that preserves (zero morphisms and) binary biproducts
    preserves binary products. -/
lemma preservesBinaryProducts_of_preservesBinaryBiproducts [PreservesBinaryBiproducts F] :
    PreservesLimitsOfShape (Discrete WalkingPair) F where
  preservesLimit {_} := preservesLimit_of_iso_diagram _ (diagramIsoPair _).symm


/-- A functor between preadditive categories that preserves (zero morphisms and) binary products
    preserves binary biproducts. -/
lemma preservesBinaryBiproduct_of_preservesBinaryProduct {X Y : C} [PreservesLimit (pair X Y) F] :
    PreservesBinaryBiproduct X Y F where
  preserves {b} hb := ⟨isBinaryBilimitOfIsLimit _ <| IsLimit.ofIsoLimit
          ((IsLimit.postcomposeHomEquiv (diagramIsoPair _) (F.mapCone b.toCone)).symm
            (isLimitOfPreserves F hb.isLimit)) <|
                      /-
                        C : Type u
                        inst✝⁵ : CategoryTheory.Category.{v, u} C
                        inst✝⁴ : CategoryTheory.Preadditive C
                        D : Type u'
                        inst✝³ : CategoryTheory.Category.{v', u'} D
                        inst✝² : CategoryTheory.Preadditive D
                        F : CategoryTheory.Functor C D
                        inst✝¹ : F.PreservesZeroMorphisms
                        X Y : C
                        inst✝ : CategoryTheory.Limits.PreservesLimit (CategoryTheory.Limits.pair X Y) F
                        b : CategoryTheory.Limits.BinaryBicone X Y
                        hb : b.IsBilimit
                        ⊢ CategoryTheory.Iso ((CategoryTheory.Limits.Cones.postcompose (CategoryTheory …
                      -/
        Cones.ext (by dsimp; rfl) fun j => by
                             /-
                               🎉 no goals
                             -/
          /-
            C : Type u
            inst✝⁵ : CategoryTheory.Category.{v, u} C
            inst✝⁴ : CategoryTheory.Preadditive C
            D : Type u'
            inst✝³ : CategoryTheory.Category.{v', u'} D
            inst✝² : CategoryTheory.Preadditive D
            F : CategoryTheory.Functor C D
            inst✝¹ : F.PreservesZeroMorphisms
            X Y : C
            inst✝ : CategoryTheory.Limits.PreservesLimit (CategoryTheory.Limits.pair X Y) F
            b : CategoryTheory.Limits.BinaryBicone X Y
            hb : b.IsBilimit
            j : CategoryTheory.Discrete CategoryTheory.Limits.WalkingPair
            ⊢ Eq (((CategoryTheory.Limits.Cones.postcompose (CategoryTheory.Limits.diagram …
          -/
                                 /-
                                   🎉 no goals
                                 -/
          rcases j with ⟨⟨⟩⟩ <;> simp⟩
                                 /-
                                   🎉 no goals
                                 -/


/-- If the (product-like) biproduct comparison for `F`, `X` and `Y` is a monomorphism, then
    `F` preserves the biproduct of `X` and `Y`. For the converse, see `map_biprod`. -/
lemma preservesBinaryBiproduct_of_mono_biprodComparison {X Y : C} [HasBinaryBiproduct X Y]
    [HasBinaryBiproduct (F.obj X) (F.obj Y)] [Mono (biprodComparison F X Y)] :
    PreservesBinaryBiproduct X Y F := by
  have that :
    prodComparison F X Y =
      (F.mapIso (biprod.isoProd X Y)).inv ≫ biprodComparison F X Y ≫ (biprod.isoProd _ _).hom := by
    ext <;> simp [← Functor.map_comp]
  /-
    C : Type u
    inst✝⁷ : CategoryTheory.Category.{v, u} C
    inst✝⁶ : CategoryTheory.Preadditive C
    D : Type u'
    inst✝⁵ : CategoryTheory.Category.{v', u'} D
    inst✝⁴ : CategoryTheory.Preadditive D
    F : CategoryTheory.Functor C D
    inst✝³ : F.PreservesZeroMorphisms
    X Y : C
    inst✝² : CategoryTheory.Limits.HasBinaryBiproduct X Y
    inst✝¹ : CategoryTheory.Limits.HasBinaryBiproduct (F.obj X) (F.obj Y)
    inst✝ : CategoryTheory.Mono (F.biprodComparison X Y)
    that : Eq (CategoryTheory.Limits.prodComparison F X Y) (CategoryTheory.Categor …
    ⊢ CategoryTheory.Limits.PreservesBinaryBiproduct X Y F
  -/
  haveI : IsIso (biprodComparison F X Y) := isIso_of_mono_of_isSplitEpi _
  haveI : IsIso (prodComparison F X Y) := by
    rw [that]
    infer_instance
  /-
    C : Type u
    inst✝⁷ : CategoryTheory.Category.{v, u} C
    inst✝⁶ : CategoryTheory.Preadditive C
    D : Type u'
    inst✝⁵ : CategoryTheory.Category.{v', u'} D
    inst✝⁴ : CategoryTheory.Preadditive D
    F : CategoryTheory.Functor C D
    inst✝³ : F.PreservesZeroMorphisms
    X Y : C
    inst✝² : CategoryTheory.Limits.HasBinaryBiproduct X Y
    inst✝¹ : CategoryTheory.Limits.HasBinaryBiproduct (F.obj X) (F.obj Y)
    inst✝ : CategoryTheory.Mono (F.biprodComparison X Y)
    that : Eq (CategoryTheory.Limits.prodComparison F X Y) (CategoryTheory.Categor …
    this✝ : CategoryTheory.IsIso (F.biprodComparison X Y)
    this : CategoryTheory.IsIso (CategoryTheory.Limits.prodComparison F X Y)
    ⊢ CategoryTheory.Limits.PreservesBinaryBiproduct X Y F
  -/
  haveI := PreservesLimitPair.of_iso_prod_comparison F X Y
  /-
    C : Type u
    inst✝⁷ : CategoryTheory.Category.{v, u} C
    inst✝⁶ : CategoryTheory.Preadditive C
    D : Type u'
    inst✝⁵ : CategoryTheory.Category.{v', u'} D
    inst✝⁴ : CategoryTheory.Preadditive D
    F : CategoryTheory.Functor C D
    inst✝³ : F.PreservesZeroMorphisms
    X Y : C
    inst✝² : CategoryTheory.Limits.HasBinaryBiproduct X Y
    inst✝¹ : CategoryTheory.Limits.HasBinaryBiproduct (F.obj X) (F.obj Y)
    inst✝ : CategoryTheory.Mono (F.biprodComparison X Y)
    that : Eq (CategoryTheory.Limits.prodComparison F X Y) (CategoryTheory.Categor …
    this✝¹ : CategoryTheory.IsIso (F.biprodComparison X Y)
    this✝ : CategoryTheory.IsIso (CategoryTheory.Limits.prodComparison F X Y)
    this : CategoryTheory.Limits.PreservesLimit (CategoryTheory.Limits.pair X Y) F
    ⊢ CategoryTheory.Limits.PreservesBinaryBiproduct X Y F
  -/
  apply preservesBinaryBiproduct_of_preservesBinaryProduct
  /-
    🎉 no goals
  -/


/-- If the (coproduct-like) biproduct comparison for `F`, `X` and `Y` is an epimorphism, then
    `F` preserves the biproduct of `X` and `Y`. For the converse, see `mapBiprod`. -/
lemma preservesBinaryBiproduct_of_epi_biprodComparison' {X Y : C} [HasBinaryBiproduct X Y]
    [HasBinaryBiproduct (F.obj X) (F.obj Y)] [Epi (biprodComparison' F X Y)] :
    PreservesBinaryBiproduct X Y F := by
  /-
    C : Type u
    inst✝⁷ : CategoryTheory.Category.{v, u} C
    inst✝⁶ : CategoryTheory.Preadditive C
    D : Type u'
    inst✝⁵ : CategoryTheory.Category.{v', u'} D
    inst✝⁴ : CategoryTheory.Preadditive D
    F : CategoryTheory.Functor C D
    inst✝³ : F.PreservesZeroMorphisms
    X Y : C
    inst✝² : CategoryTheory.Limits.HasBinaryBiproduct X Y
    inst✝¹ : CategoryTheory.Limits.HasBinaryBiproduct (F.obj X) (F.obj Y)
    inst✝ : CategoryTheory.Epi (F.biprodComparison' X Y)
    ⊢ CategoryTheory.Limits.PreservesBinaryBiproduct X Y F
  -/
  haveI : Epi (splitEpiBiprodComparison F X Y).section_ := by simpa
  haveI : IsIso (biprodComparison F X Y) :=
    IsIso.of_epi_section' (splitEpiBiprodComparison F X Y)
  /-
    C : Type u
    inst✝⁷ : CategoryTheory.Category.{v, u} C
    inst✝⁶ : CategoryTheory.Preadditive C
    D : Type u'
    inst✝⁵ : CategoryTheory.Category.{v', u'} D
    inst✝⁴ : CategoryTheory.Preadditive D
    F : CategoryTheory.Functor C D
    inst✝³ : F.PreservesZeroMorphisms
    X Y : C
    inst✝² : CategoryTheory.Limits.HasBinaryBiproduct X Y
    inst✝¹ : CategoryTheory.Limits.HasBinaryBiproduct (F.obj X) (F.obj Y)
    inst✝ : CategoryTheory.Epi (F.biprodComparison' X Y)
    this✝ : CategoryTheory.Epi (F.splitEpiBiprodComparison X Y).section_
    this : CategoryTheory.IsIso (F.biprodComparison X Y)
    ⊢ CategoryTheory.Limits.PreservesBinaryBiproduct X Y F
  -/
  apply preservesBinaryBiproduct_of_mono_biprodComparison
  /-
    🎉 no goals
  -/


/-- A functor between preadditive categories that preserves (zero morphisms and) binary products
    preserves binary biproducts. -/
lemma preservesBinaryBiproducts_of_preservesBinaryProducts
    [PreservesLimitsOfShape (Discrete WalkingPair) F] : PreservesBinaryBiproducts F where
  preserves {_} {_} := preservesBinaryBiproduct_of_preservesBinaryProduct F


/-- A functor between preadditive categories that preserves (zero morphisms and) binary biproducts
    preserves binary coproducts. -/
lemma preservesBinaryCoproduct_of_preservesBinaryBiproduct {X Y : C}
    [PreservesBinaryBiproduct X Y F] :
    PreservesColimit (pair X Y) F where
  preserves {c} hc :=
    ⟨IsColimit.ofIsoColimit
        ((IsColimit.precomposeHomEquiv (diagramIsoPair _) _).symm
          (isBinaryBilimitOfPreserves F
              (binaryBiconeIsBilimitOfColimitCoconeOfIsColimit hc)).isColimit) <|
                      /-
                        C : Type u
                        inst✝⁵ : CategoryTheory.Category.{v, u} C
                        inst✝⁴ : CategoryTheory.Preadditive C
                        D : Type u'
                        inst✝³ : CategoryTheory.Category.{v', u'} D
                        inst✝² : CategoryTheory.Preadditive D
                        F : CategoryTheory.Functor C D
                        inst✝¹ : F.PreservesZeroMorphisms
                        X Y : C
                        inst✝ : CategoryTheory.Limits.PreservesBinaryBiproduct X Y F
                        c : CategoryTheory.Limits.Cocone (CategoryTheory.Limits.pair X Y)
                        hc : CategoryTheory.Limits.IsColimit c
                        ⊢ CategoryTheory.Iso ((CategoryTheory.Limits.Cocones.precompose (CategoryTheor …
                      -/
      Cocones.ext (by dsimp; rfl) fun j => by
                             /-
                               🎉 no goals
                             -/
        /-
          C : Type u
          inst✝⁵ : CategoryTheory.Category.{v, u} C
          inst✝⁴ : CategoryTheory.Preadditive C
          D : Type u'
          inst✝³ : CategoryTheory.Category.{v', u'} D
          inst✝² : CategoryTheory.Preadditive D
          F : CategoryTheory.Functor C D
          inst✝¹ : F.PreservesZeroMorphisms
          X Y : C
          inst✝ : CategoryTheory.Limits.PreservesBinaryBiproduct X Y F
          c : CategoryTheory.Limits.Cocone (CategoryTheory.Limits.pair X Y)
          hc : CategoryTheory.Limits.IsColimit c
          j : CategoryTheory.Discrete CategoryTheory.Limits.WalkingPair
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Limits.Cocones.prec …
        -/
                               /-
                                 🎉 no goals
                               -/
        rcases j with ⟨⟨⟩⟩ <;> simp⟩
                               /-
                                 🎉 no goals
                               -/


/-- A functor between preadditive categories that preserves (zero morphisms and) binary biproducts
    preserves binary coproducts. -/
lemma preservesBinaryCoproducts_of_preservesBinaryBiproducts [PreservesBinaryBiproducts F] :
    PreservesColimitsOfShape (Discrete WalkingPair) F where
  preservesColimit {_} := preservesColimit_of_iso_diagram _ (diagramIsoPair _).symm


/-- A functor between preadditive categories that preserves (zero morphisms and) binary coproducts
    preserves binary biproducts. -/
lemma preservesBinaryBiproduct_of_preservesBinaryCoproduct {X Y : C}
    [PreservesColimit (pair X Y) F] :
    PreservesBinaryBiproduct X Y F where
  preserves {b} hb :=
    ⟨isBinaryBilimitOfIsColimit _ <|
      IsColimit.ofIsoColimit
          ((IsColimit.precomposeInvEquiv (diagramIsoPair _) (F.mapCocone b.toCocone)).symm
            (isColimitOfPreserves F hb.isColimit)) <|
        Cocones.ext (Iso.refl _) fun j => by
          /-
            C : Type u
            inst✝⁵ : CategoryTheory.Category.{v, u} C
            inst✝⁴ : CategoryTheory.Preadditive C
            D : Type u'
            inst✝³ : CategoryTheory.Category.{v', u'} D
            inst✝² : CategoryTheory.Preadditive D
            F : CategoryTheory.Functor C D
            inst✝¹ : F.PreservesZeroMorphisms
            X Y : C
            inst✝ : CategoryTheory.Limits.PreservesColimit (CategoryTheory.Limits.pair X Y …
            b : CategoryTheory.Limits.BinaryBicone X Y
            hb : b.IsBilimit
            j : CategoryTheory.Discrete CategoryTheory.Limits.WalkingPair
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Limits.Cocones.prec …
          -/
                                 /-
                                   🎉 no goals
                                 -/
          rcases j with ⟨⟨⟩⟩ <;> simp⟩
                                 /-
                                   🎉 no goals
                                 -/


/-- A functor between preadditive categories that preserves (zero morphisms and) binary coproducts
    preserves binary biproducts. -/
lemma preservesBinaryBiproducts_of_preservesBinaryCoproducts
    [PreservesColimitsOfShape (Discrete WalkingPair) F] : PreservesBinaryBiproducts F where
  preserves {_} {_} := preservesBinaryBiproduct_of_preservesBinaryCoproduct F


