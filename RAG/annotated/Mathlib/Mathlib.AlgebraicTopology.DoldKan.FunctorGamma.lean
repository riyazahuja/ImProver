/-- `Isδ₀ i` is a simple condition used to check whether a monomorphism `i` in
`SimplexCategory` identifies to the coface map `δ 0`. -/
@[nolint unusedArguments]
def Isδ₀ {Δ Δ' : SimplexCategory} (i : Δ' ⟶ Δ) [Mono i] : Prop :=
  Δ.len = Δ'.len + 1 ∧ i.toOrderHom 0 ≠ 0


theorem iff {j : ℕ} {i : Fin (j + 2)} : Isδ₀ (SimplexCategory.δ i) ↔ i = 0 := by
  /-
    j : Nat
    i : Fin (HAdd.hAdd j 2)
    ⊢ Iff (AlgebraicTopology.DoldKan.Isδ₀ (SimplexCategory.δ i)) (Eq i 0)
  -/
  constructor
    /-
      case mp
      j : Nat
      i : Fin (HAdd.hAdd j 2)
      ⊢ AlgebraicTopology.DoldKan.Isδ₀ (SimplexCategory.δ i) → Eq i 0
    -/
  · rintro ⟨_, h₂⟩
    /-
      case mp.intro
      j : Nat
      i : Fin (HAdd.hAdd j 2)
      left✝ : Eq (SimplexCategory.mk (HAdd.hAdd j 1)).len (HAdd.hAdd (SimplexCategor …
      h₂ : Ne ((SimplexCategory.Hom.toOrderHom (SimplexCategory.δ i)) 0) 0
      ⊢ Eq i 0
    -/
    by_contra h
    /-
      case mp.intro
      j : Nat
      i : Fin (HAdd.hAdd j 2)
      left✝ : Eq (SimplexCategory.mk (HAdd.hAdd j 1)).len (HAdd.hAdd (SimplexCategor …
      h₂ : Ne ((SimplexCategory.Hom.toOrderHom (SimplexCategory.δ i)) 0) 0
      h : Not (Eq i 0)
      ⊢ False
    -/
    exact h₂ (Fin.succAbove_ne_zero_zero h)
    /-
      🎉 no goals
    -/
    /-
      case mpr
      j : Nat
      i : Fin (HAdd.hAdd j 2)
      ⊢ Eq i 0 → AlgebraicTopology.DoldKan.Isδ₀ (SimplexCategory.δ i)
    -/
  · rintro rfl
    /-
      case mpr
      j : Nat
      ⊢ AlgebraicTopology.DoldKan.Isδ₀ (SimplexCategory.δ 0)
    -/
    exact ⟨rfl, by dsimp; exact Fin.succ_ne_zero (0 : Fin (j + 1))⟩
    /-
      🎉 no goals
    -/


theorem eq_δ₀ {n : ℕ} {i : ([n] : SimplexCategory) ⟶ [n + 1]} [Mono i] (hi : Isδ₀ i) :
    i = SimplexCategory.δ 0 := by
  /-
    n : Nat
    i : Quiver.Hom (SimplexCategory.mk n) (SimplexCategory.mk (HAdd.hAdd n 1))
    inst✝ : CategoryTheory.Mono i
    hi : AlgebraicTopology.DoldKan.Isδ₀ i
    ⊢ Eq i (SimplexCategory.δ 0)
  -/
  obtain ⟨j, rfl⟩ := SimplexCategory.eq_δ_of_mono i
  /-
    case intro
    n : Nat
    j : Fin (HAdd.hAdd n 2)
    inst✝ : CategoryTheory.Mono (SimplexCategory.δ j)
    hi : AlgebraicTopology.DoldKan.Isδ₀ (SimplexCategory.δ j)
    ⊢ Eq (SimplexCategory.δ j) (SimplexCategory.δ 0)
  -/
  rw [iff] at hi
  /-
    case intro
    n : Nat
    j : Fin (HAdd.hAdd n 2)
    inst✝ : CategoryTheory.Mono (SimplexCategory.δ j)
    hi : Eq j 0
    ⊢ Eq (SimplexCategory.δ j) (SimplexCategory.δ 0)
  -/
  rw [hi]
  /-
    🎉 no goals
  -/


/-- In the definition of `(Γ₀.obj K).obj Δ` as a direct sum indexed by `A : Splitting.IndexSet Δ`,
the summand `summand K Δ A` is `K.X A.1.len`. -/
def summand (Δ : SimplexCategoryᵒᵖ) (A : Splitting.IndexSet Δ) : C :=
  K.X A.1.unop.len


/-- The functor `Γ₀` sends a chain complex `K` to the simplicial object which
sends `Δ` to the direct sum of the objects `summand K Δ A` for all `A : Splitting.IndexSet Δ` -/
def obj₂ (K : ChainComplex C ℕ) (Δ : SimplexCategoryᵒᵖ) [HasFiniteCoproducts C] : C :=
  ∐ fun A : Splitting.IndexSet Δ => summand K Δ A


/-- A monomorphism `i : Δ' ⟶ Δ` induces a morphism `K.X Δ.len ⟶ K.X Δ'.len` which
is the identity if `Δ = Δ'`, the differential on the complex `K` if `i = δ 0`, and
zero otherwise. -/
def mapMono (K : ChainComplex C ℕ) {Δ' Δ : SimplexCategory} (i : Δ' ⟶ Δ) [Mono i] :
    K.X Δ.len ⟶ K.X Δ'.len := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{?u.7584, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    K✝ K' : ChainComplex C Nat
    f : Quiver.Hom K✝ K'
    Δ✝ Δ'✝ Δ'' : SimplexCategory
    K : ChainComplex C Nat
    Δ' Δ : SimplexCategory
    i : Quiver.Hom Δ' Δ
    inst✝ : CategoryTheory.Mono i
    ⊢ Quiver.Hom (K.X Δ.len) (K.X Δ'.len)
  -/
  by_cases Δ = Δ'
    /-
      case pos
      C : Type u_1
      inst✝² : CategoryTheory.Category.{?u.7584, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      K✝ K' : ChainComplex C Nat
      f : Quiver.Hom K✝ K'
      Δ✝ Δ'✝ Δ'' : SimplexCategory
      K : ChainComplex C Nat
      Δ' Δ : SimplexCategory
      i : Quiver.Hom Δ' Δ
      inst✝ : CategoryTheory.Mono i
      h✝ : Eq Δ Δ'
      ⊢ Quiver.Hom (K.X Δ.len) (K.X Δ'.len)
    -/
  · exact eqToHom (by congr)
    /-
      🎉 no goals
    -/
    /-
      case neg
      C : Type u_1
      inst✝² : CategoryTheory.Category.{?u.7584, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      K✝ K' : ChainComplex C Nat
      f : Quiver.Hom K✝ K'
      Δ✝ Δ'✝ Δ'' : SimplexCategory
      K : ChainComplex C Nat
      Δ' Δ : SimplexCategory
      i : Quiver.Hom Δ' Δ
      inst✝ : CategoryTheory.Mono i
      h✝ : Not (Eq Δ Δ')
      ⊢ Quiver.Hom (K.X Δ.len) (K.X Δ'.len)
    -/
  · by_cases Isδ₀ i
      /-
        case pos
        C : Type u_1
        inst✝² : CategoryTheory.Category.{?u.7584, u_1} C
        inst✝¹ : CategoryTheory.Preadditive C
        K✝ K' : ChainComplex C Nat
        f : Quiver.Hom K✝ K'
        Δ✝ Δ'✝ Δ'' : SimplexCategory
        K : ChainComplex C Nat
        Δ' Δ : SimplexCategory
        i : Quiver.Hom Δ' Δ
        inst✝ : CategoryTheory.Mono i
        h✝¹ : Not (Eq Δ Δ')
        h✝ : AlgebraicTopology.DoldKan.Isδ₀ i
        ⊢ Quiver.Hom (K.X Δ.len) (K.X Δ'.len)
      -/
    · exact K.d Δ.len Δ'.len
      /-
        🎉 no goals
      -/
      /-
        case neg
        C : Type u_1
        inst✝² : CategoryTheory.Category.{?u.7584, u_1} C
        inst✝¹ : CategoryTheory.Preadditive C
        K✝ K' : ChainComplex C Nat
        f : Quiver.Hom K✝ K'
        Δ✝ Δ'✝ Δ'' : SimplexCategory
        K : ChainComplex C Nat
        Δ' Δ : SimplexCategory
        i : Quiver.Hom Δ' Δ
        inst✝ : CategoryTheory.Mono i
        h✝¹ : Not (Eq Δ Δ')
        h✝ : Not (AlgebraicTopology.DoldKan.Isδ₀ i)
        ⊢ Quiver.Hom (K.X Δ.len) (K.X Δ'.len)
      -/
    · exact 0
      /-
        🎉 no goals
      -/


theorem mapMono_id : mapMono K (𝟙 Δ) = 𝟙 _ := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    K : ChainComplex C Nat
    Δ : SimplexCategory
    ⊢ Eq (AlgebraicTopology.DoldKan.Γ₀.Obj.Termwise.mapMono K (CategoryTheory.Cate …
  -/
  unfold mapMono
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    K : ChainComplex C Nat
    Δ : SimplexCategory
    ⊢ Eq (dite (Eq Δ Δ) (fun h => CategoryTheory.eqToHom ⋯) fun h => dite (Algebra …
  -/
  simp only [eq_self_iff_true, eqToHom_refl, dite_eq_ite, if_true]
  /-
    🎉 no goals
  -/


theorem mapMono_δ₀' (i : Δ' ⟶ Δ) [Mono i] (hi : Isδ₀ i) : mapMono K i = K.d Δ.len Δ'.len := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    K : ChainComplex C Nat
    Δ Δ' : SimplexCategory
    i : Quiver.Hom Δ' Δ
    inst✝ : CategoryTheory.Mono i
    hi : AlgebraicTopology.DoldKan.Isδ₀ i
    ⊢ Eq (AlgebraicTopology.DoldKan.Γ₀.Obj.Termwise.mapMono K i) (K.d Δ.len Δ'.len)
  -/
  unfold mapMono
  suffices Δ ≠ Δ' by
    simp only [dif_neg this, dif_pos hi]
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    K : ChainComplex C Nat
    Δ Δ' : SimplexCategory
    i : Quiver.Hom Δ' Δ
    inst✝ : CategoryTheory.Mono i
    hi : AlgebraicTopology.DoldKan.Isδ₀ i
    ⊢ Ne Δ Δ'
  -/
  rintro rfl
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    K : ChainComplex C Nat
    Δ : SimplexCategory
    i : Quiver.Hom Δ Δ
    inst✝ : CategoryTheory.Mono i
    hi : AlgebraicTopology.DoldKan.Isδ₀ i
    ⊢ False
  -/
  simpa only [self_eq_add_right, Nat.one_ne_zero] using hi.1
  /-
    🎉 no goals
  -/


@[simp]
theorem mapMono_δ₀ {n : ℕ} : mapMono K (δ (0 : Fin (n + 2))) = K.d (n + 1) n :=
                      /-
                        C : Type u_1
                        inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
                        inst✝ : CategoryTheory.Preadditive C
                        K : ChainComplex C Nat
                        n : Nat
                        ⊢ AlgebraicTopology.DoldKan.Isδ₀ (SimplexCategory.δ 0)
                      -/
  mapMono_δ₀' K _ (by rw [Isδ₀.iff])
                      /-
                        🎉 no goals
                      -/


theorem mapMono_eq_zero (i : Δ' ⟶ Δ) [Mono i] (h₁ : Δ ≠ Δ') (h₂ : ¬Isδ₀ i) : mapMono K i = 0 := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    K : ChainComplex C Nat
    Δ Δ' : SimplexCategory
    i : Quiver.Hom Δ' Δ
    inst✝ : CategoryTheory.Mono i
    h₁ : Ne Δ Δ'
    h₂ : Not (AlgebraicTopology.DoldKan.Isδ₀ i)
    ⊢ Eq (AlgebraicTopology.DoldKan.Γ₀.Obj.Termwise.mapMono K i) 0
  -/
  unfold mapMono
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    K : ChainComplex C Nat
    Δ Δ' : SimplexCategory
    i : Quiver.Hom Δ' Δ
    inst✝ : CategoryTheory.Mono i
    h₁ : Ne Δ Δ'
    h₂ : Not (AlgebraicTopology.DoldKan.Isδ₀ i)
    ⊢ Eq (dite (Eq Δ Δ') (fun h => CategoryTheory.eqToHom ⋯) fun h => dite (Algebr …
  -/
  rw [Ne] at h₁
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    K : ChainComplex C Nat
    Δ Δ' : SimplexCategory
    i : Quiver.Hom Δ' Δ
    inst✝ : CategoryTheory.Mono i
    h₁ : Not (Eq Δ Δ')
    h₂ : Not (AlgebraicTopology.DoldKan.Isδ₀ i)
    ⊢ Eq (dite (Eq Δ Δ') (fun h => CategoryTheory.eqToHom ⋯) fun h => dite (Algebr …
  -/
  split_ifs
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    K : ChainComplex C Nat
    Δ Δ' : SimplexCategory
    i : Quiver.Hom Δ' Δ
    inst✝ : CategoryTheory.Mono i
    h₁ : Not (Eq Δ Δ')
    h₂ : Not (AlgebraicTopology.DoldKan.Isδ₀ i)
    ⊢ Eq 0 0
  -/
  rfl
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
theorem mapMono_naturality (i : Δ ⟶ Δ') [Mono i] :
    mapMono K i ≫ f.f Δ.len = f.f Δ'.len ≫ mapMono K' i := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    K K' : ChainComplex C Nat
    f : Quiver.Hom K K'
    Δ Δ' : SimplexCategory
    i : Quiver.Hom Δ Δ'
    inst✝ : CategoryTheory.Mono i
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicTopology.DoldKan.Γ₀.Obj.Ter …
  -/
  unfold mapMono
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    K K' : ChainComplex C Nat
    f : Quiver.Hom K K'
    Δ Δ' : SimplexCategory
    i : Quiver.Hom Δ Δ'
    inst✝ : CategoryTheory.Mono i
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (dite (Eq Δ' Δ) (fun h => CategoryThe …
  -/
  split_ifs with h
    /-
      case pos
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_2, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      K K' : ChainComplex C Nat
      f : Quiver.Hom K K'
      Δ Δ' : SimplexCategory
      i : Quiver.Hom Δ Δ'
      inst✝ : CategoryTheory.Mono i
      h : Eq Δ' Δ
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom ⋯) (f.f Δ.len …
    -/
  · subst h
    /-
      case pos
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_2, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      K K' : ChainComplex C Nat
      f : Quiver.Hom K K'
      Δ' : SimplexCategory
      i : Quiver.Hom Δ' Δ'
      inst✝ : CategoryTheory.Mono i
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom ⋯) (f.f Δ'.le …
    -/
    simp only [id_comp, eqToHom_refl, comp_id]
    /-
      🎉 no goals
    -/
    /-
      case pos
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_2, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      K K' : ChainComplex C Nat
      f : Quiver.Hom K K'
      Δ Δ' : SimplexCategory
      i : Quiver.Hom Δ Δ'
      inst✝ : CategoryTheory.Mono i
      h : Not (Eq Δ' Δ)
      h✝ : AlgebraicTopology.DoldKan.Isδ₀ i
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (K.d Δ'.len Δ.len) (f.f Δ.len)) (Cate …
    -/
  · rw [HomologicalComplex.Hom.comm]
    /-
      🎉 no goals
    -/
    /-
      case neg
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_2, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      K K' : ChainComplex C Nat
      f : Quiver.Hom K K'
      Δ Δ' : SimplexCategory
      i : Quiver.Hom Δ Δ'
      inst✝ : CategoryTheory.Mono i
      h : Not (Eq Δ' Δ)
      h✝ : Not (AlgebraicTopology.DoldKan.Isδ₀ i)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp 0 (f.f Δ.len)) (CategoryTheory.Catego …
    -/
  · rw [zero_comp, comp_zero]
    /-
      🎉 no goals
    -/


@[reassoc (attr := simp)]
theorem mapMono_comp (i' : Δ'' ⟶ Δ') (i : Δ' ⟶ Δ) [Mono i'] [Mono i] :
    mapMono K i ≫ mapMono K i' = mapMono K (i' ≫ i) := by
  -- case where i : Δ' ⟶ Δ is the identity
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_2, u_1} C
    inst✝² : CategoryTheory.Preadditive C
    K : ChainComplex C Nat
    Δ Δ' Δ'' : SimplexCategory
    i' : Quiver.Hom Δ'' Δ'
    i : Quiver.Hom Δ' Δ
    inst✝¹ : CategoryTheory.Mono i'
    inst✝ : CategoryTheory.Mono i
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicTopology.DoldKan.Γ₀.Obj.Ter …
  -/
  by_cases h₁ : Δ = Δ'
    /-
      case pos
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{u_2, u_1} C
      inst✝² : CategoryTheory.Preadditive C
      K : ChainComplex C Nat
      Δ Δ' Δ'' : SimplexCategory
      i' : Quiver.Hom Δ'' Δ'
      i : Quiver.Hom Δ' Δ
      inst✝¹ : CategoryTheory.Mono i'
      inst✝ : CategoryTheory.Mono i
      h₁ : Eq Δ Δ'
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicTopology.DoldKan.Γ₀.Obj.Ter …
    -/
  · subst h₁
    /-
      case pos
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{u_2, u_1} C
      inst✝² : CategoryTheory.Preadditive C
      K : ChainComplex C Nat
      Δ Δ'' : SimplexCategory
      i' : Quiver.Hom Δ'' Δ
      i : Quiver.Hom Δ Δ
      inst✝¹ : CategoryTheory.Mono i'
      inst✝ : CategoryTheory.Mono i
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicTopology.DoldKan.Γ₀.Obj.Ter …
    -/
    simp only [SimplexCategory.eq_id_of_mono i, comp_id, id_comp, mapMono_id K, eqToHom_refl]
    /-
      🎉 no goals
    -/
  -- case where i' : Δ'' ⟶ Δ' is the identity
  /-
    case neg
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_2, u_1} C
    inst✝² : CategoryTheory.Preadditive C
    K : ChainComplex C Nat
    Δ Δ' Δ'' : SimplexCategory
    i' : Quiver.Hom Δ'' Δ'
    i : Quiver.Hom Δ' Δ
    inst✝¹ : CategoryTheory.Mono i'
    inst✝ : CategoryTheory.Mono i
    h₁ : Not (Eq Δ Δ')
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicTopology.DoldKan.Γ₀.Obj.Ter …
  -/
  by_cases h₂ : Δ' = Δ''
    /-
      case pos
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{u_2, u_1} C
      inst✝² : CategoryTheory.Preadditive C
      K : ChainComplex C Nat
      Δ Δ' Δ'' : SimplexCategory
      i' : Quiver.Hom Δ'' Δ'
      i : Quiver.Hom Δ' Δ
      inst✝¹ : CategoryTheory.Mono i'
      inst✝ : CategoryTheory.Mono i
      h₁ : Not (Eq Δ Δ')
      h₂ : Eq Δ' Δ''
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicTopology.DoldKan.Γ₀.Obj.Ter …
    -/
  · subst h₂
    /-
      case pos
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{u_2, u_1} C
      inst✝² : CategoryTheory.Preadditive C
      K : ChainComplex C Nat
      Δ Δ' : SimplexCategory
      i : Quiver.Hom Δ' Δ
      inst✝¹ : CategoryTheory.Mono i
      h₁ : Not (Eq Δ Δ')
      i' : Quiver.Hom Δ' Δ'
      inst✝ : CategoryTheory.Mono i'
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicTopology.DoldKan.Γ₀.Obj.Ter …
    -/
    simp only [SimplexCategory.eq_id_of_mono i', comp_id, id_comp, mapMono_id K, eqToHom_refl]
    /-
      🎉 no goals
    -/
  -- then the RHS is always zero
  /-
    case neg
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_2, u_1} C
    inst✝² : CategoryTheory.Preadditive C
    K : ChainComplex C Nat
    Δ Δ' Δ'' : SimplexCategory
    i' : Quiver.Hom Δ'' Δ'
    i : Quiver.Hom Δ' Δ
    inst✝¹ : CategoryTheory.Mono i'
    inst✝ : CategoryTheory.Mono i
    h₁ : Not (Eq Δ Δ')
    h₂ : Not (Eq Δ' Δ'')
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicTopology.DoldKan.Γ₀.Obj.Ter …
  -/
  obtain ⟨k, hk⟩ := Nat.exists_eq_add_of_lt (len_lt_of_mono i h₁)
  /-
    case neg.intro
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_2, u_1} C
    inst✝² : CategoryTheory.Preadditive C
    K : ChainComplex C Nat
    Δ Δ' Δ'' : SimplexCategory
    i' : Quiver.Hom Δ'' Δ'
    i : Quiver.Hom Δ' Δ
    inst✝¹ : CategoryTheory.Mono i'
    inst✝ : CategoryTheory.Mono i
    h₁ : Not (Eq Δ Δ')
    h₂ : Not (Eq Δ' Δ'')
    k : Nat
    hk : Eq Δ.len (HAdd.hAdd (HAdd.hAdd Δ'.len k) 1)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicTopology.DoldKan.Γ₀.Obj.Ter …
  -/
  obtain ⟨k', hk'⟩ := Nat.exists_eq_add_of_lt (len_lt_of_mono i' h₂)
  /-
    case neg.intro.intro
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_2, u_1} C
    inst✝² : CategoryTheory.Preadditive C
    K : ChainComplex C Nat
    Δ Δ' Δ'' : SimplexCategory
    i' : Quiver.Hom Δ'' Δ'
    i : Quiver.Hom Δ' Δ
    inst✝¹ : CategoryTheory.Mono i'
    inst✝ : CategoryTheory.Mono i
    h₁ : Not (Eq Δ Δ')
    h₂ : Not (Eq Δ' Δ'')
    k : Nat
    hk : Eq Δ.len (HAdd.hAdd (HAdd.hAdd Δ'.len k) 1)
    k' : Nat
    hk' : Eq Δ'.len (HAdd.hAdd (HAdd.hAdd Δ''.len k') 1)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicTopology.DoldKan.Γ₀.Obj.Ter …
  -/
  have eq : Δ.len = Δ''.len + (k + k' + 2) := by omega
  /-
    case neg.intro.intro
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_2, u_1} C
    inst✝² : CategoryTheory.Preadditive C
    K : ChainComplex C Nat
    Δ Δ' Δ'' : SimplexCategory
    i' : Quiver.Hom Δ'' Δ'
    i : Quiver.Hom Δ' Δ
    inst✝¹ : CategoryTheory.Mono i'
    inst✝ : CategoryTheory.Mono i
    h₁ : Not (Eq Δ Δ')
    h₂ : Not (Eq Δ' Δ'')
    k : Nat
    hk : Eq Δ.len (HAdd.hAdd (HAdd.hAdd Δ'.len k) 1)
    k' : Nat
    hk' : Eq Δ'.len (HAdd.hAdd (HAdd.hAdd Δ''.len k') 1)
    eq : Eq Δ.len (HAdd.hAdd Δ''.len (HAdd.hAdd (HAdd.hAdd k k') 2))
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicTopology.DoldKan.Γ₀.Obj.Ter …
  -/
  rw [mapMono_eq_zero K (i' ≫ i) _ _]; rotate_left
    /-
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{u_2, u_1} C
      inst✝² : CategoryTheory.Preadditive C
      K : ChainComplex C Nat
      Δ Δ' Δ'' : SimplexCategory
      i' : Quiver.Hom Δ'' Δ'
      i : Quiver.Hom Δ' Δ
      inst✝¹ : CategoryTheory.Mono i'
      inst✝ : CategoryTheory.Mono i
      h₁ : Not (Eq Δ Δ')
      h₂ : Not (Eq Δ' Δ'')
      k : Nat
      hk : Eq Δ.len (HAdd.hAdd (HAdd.hAdd Δ'.len k) 1)
      k' : Nat
      hk' : Eq Δ'.len (HAdd.hAdd (HAdd.hAdd Δ''.len k') 1)
      eq : Eq Δ.len (HAdd.hAdd Δ''.len (HAdd.hAdd (HAdd.hAdd k k') 2))
      ⊢ Ne Δ Δ''
    -/
  · by_contra h
    /-
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{u_2, u_1} C
      inst✝² : CategoryTheory.Preadditive C
      K : ChainComplex C Nat
      Δ Δ' Δ'' : SimplexCategory
      i' : Quiver.Hom Δ'' Δ'
      i : Quiver.Hom Δ' Δ
      inst✝¹ : CategoryTheory.Mono i'
      inst✝ : CategoryTheory.Mono i
      h₁ : Not (Eq Δ Δ')
      h₂ : Not (Eq Δ' Δ'')
      k : Nat
      hk : Eq Δ.len (HAdd.hAdd (HAdd.hAdd Δ'.len k) 1)
      k' : Nat
      hk' : Eq Δ'.len (HAdd.hAdd (HAdd.hAdd Δ''.len k') 1)
      eq : Eq Δ.len (HAdd.hAdd Δ''.len (HAdd.hAdd (HAdd.hAdd k k') 2))
      h : Eq Δ Δ''
      ⊢ False
    -/
    simp only [self_eq_add_right, h, add_eq_zero, and_false, reduceCtorEq] at eq
    /-
      🎉 no goals
    -/
    /-
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{u_2, u_1} C
      inst✝² : CategoryTheory.Preadditive C
      K : ChainComplex C Nat
      Δ Δ' Δ'' : SimplexCategory
      i' : Quiver.Hom Δ'' Δ'
      i : Quiver.Hom Δ' Δ
      inst✝¹ : CategoryTheory.Mono i'
      inst✝ : CategoryTheory.Mono i
      h₁ : Not (Eq Δ Δ')
      h₂ : Not (Eq Δ' Δ'')
      k : Nat
      hk : Eq Δ.len (HAdd.hAdd (HAdd.hAdd Δ'.len k) 1)
      k' : Nat
      hk' : Eq Δ'.len (HAdd.hAdd (HAdd.hAdd Δ''.len k') 1)
      eq : Eq Δ.len (HAdd.hAdd Δ''.len (HAdd.hAdd (HAdd.hAdd k k') 2))
      ⊢ Not (AlgebraicTopology.DoldKan.Isδ₀ (CategoryTheory.CategoryStruct.comp i' i))
    -/
  · by_contra h
    /-
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{u_2, u_1} C
      inst✝² : CategoryTheory.Preadditive C
      K : ChainComplex C Nat
      Δ Δ' Δ'' : SimplexCategory
      i' : Quiver.Hom Δ'' Δ'
      i : Quiver.Hom Δ' Δ
      inst✝¹ : CategoryTheory.Mono i'
      inst✝ : CategoryTheory.Mono i
      h₁ : Not (Eq Δ Δ')
      h₂ : Not (Eq Δ' Δ'')
      k : Nat
      hk : Eq Δ.len (HAdd.hAdd (HAdd.hAdd Δ'.len k) 1)
      k' : Nat
      hk' : Eq Δ'.len (HAdd.hAdd (HAdd.hAdd Δ''.len k') 1)
      eq : Eq Δ.len (HAdd.hAdd Δ''.len (HAdd.hAdd (HAdd.hAdd k k') 2))
      h : AlgebraicTopology.DoldKan.Isδ₀ (CategoryTheory.CategoryStruct.comp i' i)
      ⊢ False
    -/
    simp only [h.1, add_right_inj] at eq
    /-
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{u_2, u_1} C
      inst✝² : CategoryTheory.Preadditive C
      K : ChainComplex C Nat
      Δ Δ' Δ'' : SimplexCategory
      i' : Quiver.Hom Δ'' Δ'
      i : Quiver.Hom Δ' Δ
      inst✝¹ : CategoryTheory.Mono i'
      inst✝ : CategoryTheory.Mono i
      h₁ : Not (Eq Δ Δ')
      h₂ : Not (Eq Δ' Δ'')
      k : Nat
      hk : Eq Δ.len (HAdd.hAdd (HAdd.hAdd Δ'.len k) 1)
      k' : Nat
      hk' : Eq Δ'.len (HAdd.hAdd (HAdd.hAdd Δ''.len k') 1)
      h : AlgebraicTopology.DoldKan.Isδ₀ (CategoryTheory.CategoryStruct.comp i' i)
      eq : Eq 1 (HAdd.hAdd (HAdd.hAdd k k') 2)
      ⊢ False
    -/
    omega
    /-
      🎉 no goals
    -/
  -- in all cases, the LHS is also zero, either by definition, or because d ≫ d = 0
  /-
    case neg.intro.intro
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_2, u_1} C
    inst✝² : CategoryTheory.Preadditive C
    K : ChainComplex C Nat
    Δ Δ' Δ'' : SimplexCategory
    i' : Quiver.Hom Δ'' Δ'
    i : Quiver.Hom Δ' Δ
    inst✝¹ : CategoryTheory.Mono i'
    inst✝ : CategoryTheory.Mono i
    h₁ : Not (Eq Δ Δ')
    h₂ : Not (Eq Δ' Δ'')
    k : Nat
    hk : Eq Δ.len (HAdd.hAdd (HAdd.hAdd Δ'.len k) 1)
    k' : Nat
    hk' : Eq Δ'.len (HAdd.hAdd (HAdd.hAdd Δ''.len k') 1)
    eq : Eq Δ.len (HAdd.hAdd Δ''.len (HAdd.hAdd (HAdd.hAdd k k') 2))
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicTopology.DoldKan.Γ₀.Obj.Ter …
  -/
  by_cases h₃ : Isδ₀ i
    /-
      case pos
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{u_2, u_1} C
      inst✝² : CategoryTheory.Preadditive C
      K : ChainComplex C Nat
      Δ Δ' Δ'' : SimplexCategory
      i' : Quiver.Hom Δ'' Δ'
      i : Quiver.Hom Δ' Δ
      inst✝¹ : CategoryTheory.Mono i'
      inst✝ : CategoryTheory.Mono i
      h₁ : Not (Eq Δ Δ')
      h₂ : Not (Eq Δ' Δ'')
      k : Nat
      hk : Eq Δ.len (HAdd.hAdd (HAdd.hAdd Δ'.len k) 1)
      k' : Nat
      hk' : Eq Δ'.len (HAdd.hAdd (HAdd.hAdd Δ''.len k') 1)
      eq : Eq Δ.len (HAdd.hAdd Δ''.len (HAdd.hAdd (HAdd.hAdd k k') 2))
      h₃ : AlgebraicTopology.DoldKan.Isδ₀ i
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicTopology.DoldKan.Γ₀.Obj.Ter …
    -/
  · by_cases h₄ : Isδ₀ i'
      /-
        case pos
        C : Type u_1
        inst✝³ : CategoryTheory.Category.{u_2, u_1} C
        inst✝² : CategoryTheory.Preadditive C
        K : ChainComplex C Nat
        Δ Δ' Δ'' : SimplexCategory
        i' : Quiver.Hom Δ'' Δ'
        i : Quiver.Hom Δ' Δ
        inst✝¹ : CategoryTheory.Mono i'
        inst✝ : CategoryTheory.Mono i
        h₁ : Not (Eq Δ Δ')
        h₂ : Not (Eq Δ' Δ'')
        k : Nat
        hk : Eq Δ.len (HAdd.hAdd (HAdd.hAdd Δ'.len k) 1)
        k' : Nat
        hk' : Eq Δ'.len (HAdd.hAdd (HAdd.hAdd Δ''.len k') 1)
        eq : Eq Δ.len (HAdd.hAdd Δ''.len (HAdd.hAdd (HAdd.hAdd k k') 2))
        h₃ : AlgebraicTopology.DoldKan.Isδ₀ i
        h₄ : AlgebraicTopology.DoldKan.Isδ₀ i'
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicTopology.DoldKan.Γ₀.Obj.Ter …
      -/
    · rw [mapMono_δ₀' K i h₃, mapMono_δ₀' K i' h₄, HomologicalComplex.d_comp_d]
      /-
        🎉 no goals
      -/
      /-
        case neg
        C : Type u_1
        inst✝³ : CategoryTheory.Category.{u_2, u_1} C
        inst✝² : CategoryTheory.Preadditive C
        K : ChainComplex C Nat
        Δ Δ' Δ'' : SimplexCategory
        i' : Quiver.Hom Δ'' Δ'
        i : Quiver.Hom Δ' Δ
        inst✝¹ : CategoryTheory.Mono i'
        inst✝ : CategoryTheory.Mono i
        h₁ : Not (Eq Δ Δ')
        h₂ : Not (Eq Δ' Δ'')
        k : Nat
        hk : Eq Δ.len (HAdd.hAdd (HAdd.hAdd Δ'.len k) 1)
        k' : Nat
        hk' : Eq Δ'.len (HAdd.hAdd (HAdd.hAdd Δ''.len k') 1)
        eq : Eq Δ.len (HAdd.hAdd Δ''.len (HAdd.hAdd (HAdd.hAdd k k') 2))
        h₃ : AlgebraicTopology.DoldKan.Isδ₀ i
        h₄ : Not (AlgebraicTopology.DoldKan.Isδ₀ i')
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicTopology.DoldKan.Γ₀.Obj.Ter …
      -/
    · simp only [mapMono_eq_zero K i' h₂ h₄, comp_zero]
      /-
        🎉 no goals
      -/
    /-
      case neg
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{u_2, u_1} C
      inst✝² : CategoryTheory.Preadditive C
      K : ChainComplex C Nat
      Δ Δ' Δ'' : SimplexCategory
      i' : Quiver.Hom Δ'' Δ'
      i : Quiver.Hom Δ' Δ
      inst✝¹ : CategoryTheory.Mono i'
      inst✝ : CategoryTheory.Mono i
      h₁ : Not (Eq Δ Δ')
      h₂ : Not (Eq Δ' Δ'')
      k : Nat
      hk : Eq Δ.len (HAdd.hAdd (HAdd.hAdd Δ'.len k) 1)
      k' : Nat
      hk' : Eq Δ'.len (HAdd.hAdd (HAdd.hAdd Δ''.len k') 1)
      eq : Eq Δ.len (HAdd.hAdd Δ''.len (HAdd.hAdd (HAdd.hAdd k k') 2))
      h₃ : Not (AlgebraicTopology.DoldKan.Isδ₀ i)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicTopology.DoldKan.Γ₀.Obj.Ter …
    -/
  · simp only [mapMono_eq_zero K i h₁ h₃, zero_comp]
    /-
      🎉 no goals
    -/


/-- The simplicial morphism on the simplicial object `Γ₀.obj K` induced by
a morphism `Δ' → Δ` in `SimplexCategory` is defined on each summand
associated to an `A : Splitting.IndexSet Δ` in terms of the epi-mono factorisation
of `θ ≫ A.e`. -/
def map (K : ChainComplex C ℕ) {Δ' Δ : SimplexCategoryᵒᵖ} (θ : Δ ⟶ Δ') : obj₂ K Δ ⟶ obj₂ K Δ' :=
  Sigma.desc fun A =>
    Termwise.mapMono K (image.ι (θ.unop ≫ A.e)) ≫ Sigma.ι (summand K Δ') (A.pull θ)


@[reassoc]
theorem map_on_summand₀ {Δ Δ' : SimplexCategoryᵒᵖ} (A : Splitting.IndexSet Δ) {θ : Δ ⟶ Δ'}
    {Δ'' : SimplexCategory} {e : Δ'.unop ⟶ Δ''} {i : Δ'' ⟶ A.1.unop} [Epi e] [Mono i]
    (fac : e ≫ i = θ.unop ≫ A.e) :
    Sigma.ι (summand K Δ) A ≫ map K θ =
      Termwise.mapMono K i ≫ Sigma.ι (summand K Δ') (Splitting.IndexSet.mk e) := by
  /-
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_2, u_1} C
    inst✝³ : CategoryTheory.Preadditive C
    K : ChainComplex C Nat
    inst✝² : CategoryTheory.Limits.HasFiniteCoproducts C
    Δ Δ' : Opposite SimplexCategory
    A : SimplicialObject.Splitting.IndexSet Δ
    θ : Quiver.Hom Δ Δ'
    Δ'' : SimplexCategory
    e : Quiver.Hom (Opposite.unop Δ') Δ''
    i : Quiver.Hom Δ'' (Opposite.unop A.fst)
    inst✝¹ : CategoryTheory.Epi e
    inst✝ : CategoryTheory.Mono i
    fac : Eq (CategoryTheory.CategoryStruct.comp e i) (CategoryTheory.CategoryStru …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Sigma.ι (Algeb …
  -/
  simp only [map, colimit.ι_desc, Cofan.mk_ι_app]
  /-
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_2, u_1} C
    inst✝³ : CategoryTheory.Preadditive C
    K : ChainComplex C Nat
    inst✝² : CategoryTheory.Limits.HasFiniteCoproducts C
    Δ Δ' : Opposite SimplexCategory
    A : SimplicialObject.Splitting.IndexSet Δ
    θ : Quiver.Hom Δ Δ'
    Δ'' : SimplexCategory
    e : Quiver.Hom (Opposite.unop Δ') Δ''
    i : Quiver.Hom Δ'' (Opposite.unop A.fst)
    inst✝¹ : CategoryTheory.Epi e
    inst✝ : CategoryTheory.Mono i
    fac : Eq (CategoryTheory.CategoryStruct.comp e i) (CategoryTheory.CategoryStru …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicTopology.DoldKan.Γ₀.Obj.Ter …
  -/
  have h := SimplexCategory.image_eq fac
  /-
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_2, u_1} C
    inst✝³ : CategoryTheory.Preadditive C
    K : ChainComplex C Nat
    inst✝² : CategoryTheory.Limits.HasFiniteCoproducts C
    Δ Δ' : Opposite SimplexCategory
    A : SimplicialObject.Splitting.IndexSet Δ
    θ : Quiver.Hom Δ Δ'
    Δ'' : SimplexCategory
    e : Quiver.Hom (Opposite.unop Δ') Δ''
    i : Quiver.Hom Δ'' (Opposite.unop A.fst)
    inst✝¹ : CategoryTheory.Epi e
    inst✝ : CategoryTheory.Mono i
    fac : Eq (CategoryTheory.CategoryStruct.comp e i) (CategoryTheory.CategoryStru …
    h : Eq (CategoryTheory.Limits.image (CategoryTheory.CategoryStruct.comp θ.unop …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicTopology.DoldKan.Γ₀.Obj.Ter …
  -/
  subst h
  /-
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_2, u_1} C
    inst✝³ : CategoryTheory.Preadditive C
    K : ChainComplex C Nat
    inst✝² : CategoryTheory.Limits.HasFiniteCoproducts C
    Δ Δ' : Opposite SimplexCategory
    A : SimplicialObject.Splitting.IndexSet Δ
    θ : Quiver.Hom Δ Δ'
    e : Quiver.Hom (Opposite.unop Δ') (CategoryTheory.Limits.image (CategoryTheory …
    i : Quiver.Hom (CategoryTheory.Limits.image (CategoryTheory.CategoryStruct.com …
    inst✝¹ : CategoryTheory.Epi e
    inst✝ : CategoryTheory.Mono i
    fac : Eq (CategoryTheory.CategoryStruct.comp e i) (CategoryTheory.CategoryStru …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicTopology.DoldKan.Γ₀.Obj.Ter …
  -/
  congr
    /-
      case e_a.e_i
      C : Type u_1
      inst✝⁴ : CategoryTheory.Category.{u_2, u_1} C
      inst✝³ : CategoryTheory.Preadditive C
      K : ChainComplex C Nat
      inst✝² : CategoryTheory.Limits.HasFiniteCoproducts C
      Δ Δ' : Opposite SimplexCategory
      A : SimplicialObject.Splitting.IndexSet Δ
      θ : Quiver.Hom Δ Δ'
      e : Quiver.Hom (Opposite.unop Δ') (CategoryTheory.Limits.image (CategoryTheory …
      i : Quiver.Hom (CategoryTheory.Limits.image (CategoryTheory.CategoryStruct.com …
      inst✝¹ : CategoryTheory.Epi e
      inst✝ : CategoryTheory.Mono i
      fac : Eq (CategoryTheory.CategoryStruct.comp e i) (CategoryTheory.CategoryStru …
      ⊢ Eq (CategoryTheory.Limits.image.ι (CategoryTheory.CategoryStruct.comp θ.unop …
    -/
  · exact SimplexCategory.image_ι_eq fac
    /-
      🎉 no goals
    -/
    /-
      case e_a.h.e_6.h
      C : Type u_1
      inst✝⁴ : CategoryTheory.Category.{u_2, u_1} C
      inst✝³ : CategoryTheory.Preadditive C
      K : ChainComplex C Nat
      inst✝² : CategoryTheory.Limits.HasFiniteCoproducts C
      Δ Δ' : Opposite SimplexCategory
      A : SimplicialObject.Splitting.IndexSet Δ
      θ : Quiver.Hom Δ Δ'
      e : Quiver.Hom (Opposite.unop Δ') (CategoryTheory.Limits.image (CategoryTheory …
      i : Quiver.Hom (CategoryTheory.Limits.image (CategoryTheory.CategoryStruct.com …
      inst✝¹ : CategoryTheory.Epi e
      inst✝ : CategoryTheory.Mono i
      fac : Eq (CategoryTheory.CategoryStruct.comp e i) (CategoryTheory.CategoryStru …
      ⊢ Eq (A.pull θ) (SimplicialObject.Splitting.IndexSet.mk e)
    -/
  · dsimp only [SimplicialObject.Splitting.IndexSet.pull]
    /-
      case e_a.h.e_6.h
      C : Type u_1
      inst✝⁴ : CategoryTheory.Category.{u_2, u_1} C
      inst✝³ : CategoryTheory.Preadditive C
      K : ChainComplex C Nat
      inst✝² : CategoryTheory.Limits.HasFiniteCoproducts C
      Δ Δ' : Opposite SimplexCategory
      A : SimplicialObject.Splitting.IndexSet Δ
      θ : Quiver.Hom Δ Δ'
      e : Quiver.Hom (Opposite.unop Δ') (CategoryTheory.Limits.image (CategoryTheory …
      i : Quiver.Hom (CategoryTheory.Limits.image (CategoryTheory.CategoryStruct.com …
      inst✝¹ : CategoryTheory.Epi e
      inst✝ : CategoryTheory.Mono i
      fac : Eq (CategoryTheory.CategoryStruct.comp e i) (CategoryTheory.CategoryStru …
      ⊢ Eq (SimplicialObject.Splitting.IndexSet.mk (CategoryTheory.Limits.factorThru …
    -/
    congr
    /-
      case e_a.h.e_6.h.e_f
      C : Type u_1
      inst✝⁴ : CategoryTheory.Category.{u_2, u_1} C
      inst✝³ : CategoryTheory.Preadditive C
      K : ChainComplex C Nat
      inst✝² : CategoryTheory.Limits.HasFiniteCoproducts C
      Δ Δ' : Opposite SimplexCategory
      A : SimplicialObject.Splitting.IndexSet Δ
      θ : Quiver.Hom Δ Δ'
      e : Quiver.Hom (Opposite.unop Δ') (CategoryTheory.Limits.image (CategoryTheory …
      i : Quiver.Hom (CategoryTheory.Limits.image (CategoryTheory.CategoryStruct.com …
      inst✝¹ : CategoryTheory.Epi e
      inst✝ : CategoryTheory.Mono i
      fac : Eq (CategoryTheory.CategoryStruct.comp e i) (CategoryTheory.CategoryStru …
      ⊢ Eq (CategoryTheory.Limits.factorThruImage (CategoryTheory.CategoryStruct.com …
    -/
    exact SimplexCategory.factorThruImage_eq fac
    /-
      🎉 no goals
    -/


@[reassoc]
theorem map_on_summand₀' {Δ Δ' : SimplexCategoryᵒᵖ} (A : Splitting.IndexSet Δ) (θ : Δ ⟶ Δ') :
    Sigma.ι (summand K Δ) A ≫ map K θ =
      Termwise.mapMono K (image.ι (θ.unop ≫ A.e)) ≫ Sigma.ι (summand K _) (A.pull θ) :=
  map_on_summand₀ K A (A.fac_pull θ)


/-- The functor `Γ₀ : ChainComplex C ℕ ⥤ SimplicialObject C`, on objects. -/
@[simps]
def obj (K : ChainComplex C ℕ) : SimplicialObject C where
  obj Δ := Obj.obj₂ K Δ
  map θ := Obj.map K θ
  map_id Δ := colimit.hom_ext (fun ⟨A⟩ => by
    /-
      C : Type u_1
      inst✝² : CategoryTheory.Category.{?u.43695, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      K✝ K' : ChainComplex C Nat
      f : Quiver.Hom K✝ K'
      Δ✝ Δ' Δ'' : SimplexCategory
      inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
      K : ChainComplex C Nat
      Δ : Opposite SimplexCategory
      x✝ : CategoryTheory.Discrete (SimplicialObject.Splitting.IndexSet Δ)
      A : SimplicialObject.Splitting.IndexSet Δ
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.ι (Cat …
    -/
    dsimp
    /-
      C : Type u_1
      inst✝² : CategoryTheory.Category.{?u.43695, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      K✝ K' : ChainComplex C Nat
      f : Quiver.Hom K✝ K'
      Δ✝ Δ' Δ'' : SimplexCategory
      inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
      K : ChainComplex C Nat
      Δ : Opposite SimplexCategory
      x✝ : CategoryTheory.Discrete (SimplicialObject.Splitting.IndexSet Δ)
      A : SimplicialObject.Splitting.IndexSet Δ
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.ι (Cat …
    -/
    have fac : A.e ≫ 𝟙 A.1.unop = (𝟙 Δ).unop ≫ A.e := by rw [unop_id, comp_id, id_comp]
    /-
      C : Type u_1
      inst✝² : CategoryTheory.Category.{?u.43695, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      K✝ K' : ChainComplex C Nat
      f : Quiver.Hom K✝ K'
      Δ✝ Δ' Δ'' : SimplexCategory
      inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
      K : ChainComplex C Nat
      Δ : Opposite SimplexCategory
      x✝ : CategoryTheory.Discrete (SimplicialObject.Splitting.IndexSet Δ)
      A : SimplicialObject.Splitting.IndexSet Δ
      fac : Eq (CategoryTheory.CategoryStruct.comp A.e (CategoryTheory.CategoryStruc …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.ι (Cat …
    -/
    erw [Obj.map_on_summand₀ K A fac, Obj.Termwise.mapMono_id, id_comp, comp_id]
    /-
      C : Type u_1
      inst✝² : CategoryTheory.Category.{?u.43695, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      K✝ K' : ChainComplex C Nat
      f : Quiver.Hom K✝ K'
      Δ✝ Δ' Δ'' : SimplexCategory
      inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
      K : ChainComplex C Nat
      Δ : Opposite SimplexCategory
      x✝ : CategoryTheory.Discrete (SimplicialObject.Splitting.IndexSet Δ)
      A : SimplicialObject.Splitting.IndexSet Δ
      fac : Eq (CategoryTheory.CategoryStruct.comp A.e (CategoryTheory.CategoryStruc …
      ⊢ Eq (CategoryTheory.Limits.Sigma.ι (AlgebraicTopology.DoldKan.Γ₀.Obj.summand  …
    -/
    rfl)
    /-
      🎉 no goals
    -/
  map_comp {Δ'' Δ' Δ} θ' θ := colimit.hom_ext (fun ⟨A⟩ => by
    /-
      C : Type u_1
      inst✝² : CategoryTheory.Category.{?u.43695, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      K✝ K' : ChainComplex C Nat
      f : Quiver.Hom K✝ K'
      Δ✝ Δ'✝ Δ''✝ : SimplexCategory
      inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
      K : ChainComplex C Nat
      Δ'' Δ' Δ : Opposite SimplexCategory
      θ' : Quiver.Hom Δ'' Δ'
      θ : Quiver.Hom Δ' Δ
      x✝ : CategoryTheory.Discrete (SimplicialObject.Splitting.IndexSet Δ'')
      A : SimplicialObject.Splitting.IndexSet Δ''
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.ι (Cat …
    -/
    have fac : θ.unop ≫ θ'.unop ≫ A.e = (θ' ≫ θ).unop ≫ A.e := by rw [unop_comp, assoc]
    rw [← image.fac (θ'.unop ≫ A.e), ← assoc, ←
      image.fac (θ.unop ≫ factorThruImage (θ'.unop ≫ A.e)), assoc] at fac
    simp only [Obj.map_on_summand₀'_assoc K A θ', Obj.map_on_summand₀' K _ θ,
      Obj.Termwise.mapMono_comp_assoc, Obj.map_on_summand₀ K A fac]
    /-
      C : Type u_1
      inst✝² : CategoryTheory.Category.{?u.43695, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      K✝ K' : ChainComplex C Nat
      f : Quiver.Hom K✝ K'
      Δ✝ Δ'✝ Δ''✝ : SimplexCategory
      inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
      K : ChainComplex C Nat
      Δ'' Δ' Δ : Opposite SimplexCategory
      θ' : Quiver.Hom Δ'' Δ'
      θ : Quiver.Hom Δ' Δ
      x✝ : CategoryTheory.Discrete (SimplicialObject.Splitting.IndexSet Δ'')
      A : SimplicialObject.Splitting.IndexSet Δ''
      fac : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.factorThru …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicTopology.DoldKan.Γ₀.Obj.Ter …
    -/
    rfl)
    /-
      🎉 no goals
    -/


/-- By construction, the simplicial `Γ₀.obj K` is equipped with a splitting. -/
def splitting (K : ChainComplex C ℕ) : SimplicialObject.Splitting (Γ₀.obj K) where
  N n := K.X n
  ι n := Sigma.ι (Γ₀.Obj.summand K (op [n])) (Splitting.IndexSet.id (op [n]))
  isColimit' Δ := IsColimit.ofIsoColimit (colimit.isColimit _) (Cofan.ext (Iso.refl _) (by
      /-
        C : Type u_1
        inst✝² : CategoryTheory.Category.{?u.56612, u_1} C
        inst✝¹ : CategoryTheory.Preadditive C
        K✝ K' : ChainComplex C Nat
        f : Quiver.Hom K✝ K'
        Δ✝ Δ' Δ'' : SimplexCategory
        inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
        K : ChainComplex C Nat
        Δ : Opposite SimplexCategory
        ⊢ ∀ (b : SimplicialObject.Splitting.IndexSet Δ), Eq (CategoryTheory.CategorySt …
      -/
      intro A
      /-
        C : Type u_1
        inst✝² : CategoryTheory.Category.{?u.56612, u_1} C
        inst✝¹ : CategoryTheory.Preadditive C
        K✝ K' : ChainComplex C Nat
        f : Quiver.Hom K✝ K'
        Δ✝ Δ' Δ'' : SimplexCategory
        inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
        K : ChainComplex C Nat
        Δ : Opposite SimplexCategory
        A : SimplicialObject.Splitting.IndexSet Δ
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofan.inj (Cat …
      -/
      dsimp [Splitting.cofan']
      rw [comp_id, Γ₀.Obj.map_on_summand₀ K (SimplicialObject.Splitting.IndexSet.id A.1)
        (show A.e ≫ 𝟙 _ = A.e.op.unop ≫ 𝟙 _ by rfl), Γ₀.Obj.Termwise.mapMono_id]
      /-
        C : Type u_1
        inst✝² : CategoryTheory.Category.{?u.56612, u_1} C
        inst✝¹ : CategoryTheory.Preadditive C
        K✝ K' : ChainComplex C Nat
        f : Quiver.Hom K✝ K'
        Δ✝ Δ' Δ'' : SimplexCategory
        inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
        K : ChainComplex C Nat
        Δ : Opposite SimplexCategory
        A : SimplicialObject.Splitting.IndexSet Δ
        ⊢ Eq (CategoryTheory.Limits.Cofan.inj (CategoryTheory.Limits.colimit.cocone (C …
      -/
      dsimp
      /-
        C : Type u_1
        inst✝² : CategoryTheory.Category.{?u.56612, u_1} C
        inst✝¹ : CategoryTheory.Preadditive C
        K✝ K' : ChainComplex C Nat
        f : Quiver.Hom K✝ K'
        Δ✝ Δ' Δ'' : SimplexCategory
        inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
        K : ChainComplex C Nat
        Δ : Opposite SimplexCategory
        A : SimplicialObject.Splitting.IndexSet Δ
        ⊢ Eq (CategoryTheory.Limits.Cofan.inj (CategoryTheory.Limits.colimit.cocone (C …
      -/
      rw [id_comp]
      /-
        C : Type u_1
        inst✝² : CategoryTheory.Category.{?u.56612, u_1} C
        inst✝¹ : CategoryTheory.Preadditive C
        K✝ K' : ChainComplex C Nat
        f : Quiver.Hom K✝ K'
        Δ✝ Δ' Δ'' : SimplexCategory
        inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
        K : ChainComplex C Nat
        Δ : Opposite SimplexCategory
        A : SimplicialObject.Splitting.IndexSet Δ
        ⊢ Eq (CategoryTheory.Limits.Cofan.inj (CategoryTheory.Limits.colimit.cocone (C …
      -/
      rfl))
      /-
        🎉 no goals
      -/


@[reassoc]
theorem Obj.map_on_summand {Δ Δ' : SimplexCategoryᵒᵖ} (A : Splitting.IndexSet Δ) (θ : Δ ⟶ Δ')
    {Δ'' : SimplexCategory} {e : Δ'.unop ⟶ Δ''} {i : Δ'' ⟶ A.1.unop} [Epi e] [Mono i]
    (fac : e ≫ i = θ.unop ≫ A.e) :
    ((Γ₀.splitting K).cofan Δ).inj A ≫ (Γ₀.obj K).map θ =
      Γ₀.Obj.Termwise.mapMono K i ≫ ((Γ₀.splitting K).cofan Δ').inj (Splitting.IndexSet.mk e) := by
  /-
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_2, u_1} C
    inst✝³ : CategoryTheory.Preadditive C
    K : ChainComplex C Nat
    inst✝² : CategoryTheory.Limits.HasFiniteCoproducts C
    Δ Δ' : Opposite SimplexCategory
    A : SimplicialObject.Splitting.IndexSet Δ
    θ : Quiver.Hom Δ Δ'
    Δ'' : SimplexCategory
    e : Quiver.Hom (Opposite.unop Δ') Δ''
    i : Quiver.Hom Δ'' (Opposite.unop A.fst)
    inst✝¹ : CategoryTheory.Epi e
    inst✝ : CategoryTheory.Mono i
    fac : Eq (CategoryTheory.CategoryStruct.comp e i) (CategoryTheory.CategoryStru …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (((AlgebraicTopology.DoldKan.Γ₀.split …
  -/
  dsimp [Splitting.cofan]
  /-
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_2, u_1} C
    inst✝³ : CategoryTheory.Preadditive C
    K : ChainComplex C Nat
    inst✝² : CategoryTheory.Limits.HasFiniteCoproducts C
    Δ Δ' : Opposite SimplexCategory
    A : SimplicialObject.Splitting.IndexSet Δ
    θ : Quiver.Hom Δ Δ'
    Δ'' : SimplexCategory
    e : Quiver.Hom (Opposite.unop Δ') Δ''
    i : Quiver.Hom Δ'' (Opposite.unop A.fst)
    inst✝¹ : CategoryTheory.Epi e
    inst✝ : CategoryTheory.Mono i
    fac : Eq (CategoryTheory.CategoryStruct.comp e i) (CategoryTheory.CategoryStru …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  change (_ ≫ (Γ₀.obj K).map A.e.op) ≫ (Γ₀.obj K).map θ = _
  /-
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_2, u_1} C
    inst✝³ : CategoryTheory.Preadditive C
    K : ChainComplex C Nat
    inst✝² : CategoryTheory.Limits.HasFiniteCoproducts C
    Δ Δ' : Opposite SimplexCategory
    A : SimplicialObject.Splitting.IndexSet Δ
    θ : Quiver.Hom Δ Δ'
    Δ'' : SimplexCategory
    e : Quiver.Hom (Opposite.unop Δ') Δ''
    i : Quiver.Hom Δ'' (Opposite.unop A.fst)
    inst✝¹ : CategoryTheory.Epi e
    inst✝ : CategoryTheory.Mono i
    fac : Eq (CategoryTheory.CategoryStruct.comp e i) (CategoryTheory.CategoryStru …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  rw [assoc, ← Functor.map_comp]
  /-
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_2, u_1} C
    inst✝³ : CategoryTheory.Preadditive C
    K : ChainComplex C Nat
    inst✝² : CategoryTheory.Limits.HasFiniteCoproducts C
    Δ Δ' : Opposite SimplexCategory
    A : SimplicialObject.Splitting.IndexSet Δ
    θ : Quiver.Hom Δ Δ'
    Δ'' : SimplexCategory
    e : Quiver.Hom (Opposite.unop Δ') Δ''
    i : Quiver.Hom Δ'' (Opposite.unop A.fst)
    inst✝¹ : CategoryTheory.Epi e
    inst✝ : CategoryTheory.Mono i
    fac : Eq (CategoryTheory.CategoryStruct.comp e i) (CategoryTheory.CategoryStru …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((AlgebraicTopology.DoldKan.Γ₀.splitt …
  -/
  dsimp [splitting]
  erw [Γ₀.Obj.map_on_summand₀ K (Splitting.IndexSet.id A.1)
    (show e ≫ i = ((Splitting.IndexSet.e A).op ≫ θ).unop ≫ 𝟙 _ by rw [comp_id, fac]; rfl),
    Γ₀.Obj.map_on_summand₀ K (Splitting.IndexSet.id (op Δ''))
      (show e ≫ 𝟙 Δ'' = e.op.unop ≫ 𝟙 _ by simp), Termwise.mapMono_id, id_comp]


@[reassoc]
theorem Obj.map_on_summand' {Δ Δ' : SimplexCategoryᵒᵖ} (A : Splitting.IndexSet Δ) (θ : Δ ⟶ Δ') :
    ((splitting K).cofan Δ).inj A ≫ (obj K).map θ =
      Obj.Termwise.mapMono K (image.ι (θ.unop ≫ A.e)) ≫
        ((splitting K).cofan Δ').inj (A.pull θ) := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    K : ChainComplex C Nat
    inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
    Δ Δ' : Opposite SimplexCategory
    A : SimplicialObject.Splitting.IndexSet Δ
    θ : Quiver.Hom Δ Δ'
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (((AlgebraicTopology.DoldKan.Γ₀.split …
  -/
  apply Obj.map_on_summand
  /-
    case fac
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    K : ChainComplex C Nat
    inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
    Δ Δ' : Opposite SimplexCategory
    A : SimplicialObject.Splitting.IndexSet Δ
    θ : Quiver.Hom Δ Δ'
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.factorThruImag …
  -/
  apply image.fac
  /-
    🎉 no goals
  -/


@[reassoc]
theorem Obj.mapMono_on_summand_id {Δ Δ' : SimplexCategory} (i : Δ' ⟶ Δ) [Mono i] :
    ((splitting K).cofan _).inj (Splitting.IndexSet.id (op Δ)) ≫ (obj K).map i.op =
      Obj.Termwise.mapMono K i ≫ ((splitting K).cofan _).inj (Splitting.IndexSet.id (op Δ')) :=
  Obj.map_on_summand K (Splitting.IndexSet.id (op Δ)) i.op (rfl : 𝟙 _ ≫ i = i ≫ 𝟙 _)


@[reassoc]
theorem Obj.map_epi_on_summand_id {Δ Δ' : SimplexCategory} (e : Δ' ⟶ Δ) [Epi e] :
    ((Γ₀.splitting K).cofan _).inj (Splitting.IndexSet.id (op Δ)) ≫ (Γ₀.obj K).map e.op =
      ((Γ₀.splitting K).cofan _).inj (Splitting.IndexSet.mk e) := by
  simpa only [Γ₀.Obj.map_on_summand K (Splitting.IndexSet.id (op Δ)) e.op
      (rfl : e ≫ 𝟙 Δ = e ≫ 𝟙 Δ),
    Γ₀.Obj.Termwise.mapMono_id] using id_comp _


/-- The functor `Γ₀ : ChainComplex C ℕ ⥤ SimplicialObject C`, on morphisms. -/
@[simps]
def map {K K' : ChainComplex C ℕ} (f : K ⟶ K') : obj K ⟶ obj K' where
  app Δ := (Γ₀.splitting K).desc Δ fun A => f.f A.1.unop.len ≫
    ((Γ₀.splitting K').cofan _).inj A
  naturality {Δ' Δ} θ := by
    /-
      C : Type u_1
      inst✝² : CategoryTheory.Category.{?u.79960, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      K✝ K'✝ : ChainComplex C Nat
      f✝ : Quiver.Hom K✝ K'✝
      Δ✝ Δ'✝ Δ'' : SimplexCategory
      inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
      K K' : ChainComplex C Nat
      f : Quiver.Hom K K'
      Δ' Δ : Opposite SimplexCategory
      θ : Quiver.Hom Δ' Δ
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((AlgebraicTopology.DoldKan.Γ₀.obj K) …
    -/
    apply (Γ₀.splitting K).hom_ext'
    /-
      case h
      C : Type u_1
      inst✝² : CategoryTheory.Category.{?u.79960, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      K✝ K'✝ : ChainComplex C Nat
      f✝ : Quiver.Hom K✝ K'✝
      Δ✝ Δ'✝ Δ'' : SimplexCategory
      inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
      K K' : ChainComplex C Nat
      f : Quiver.Hom K K'
      Δ' Δ : Opposite SimplexCategory
      θ : Quiver.Hom Δ' Δ
      ⊢ ∀ (A : SimplicialObject.Splitting.IndexSet Δ'), Eq (CategoryTheory.CategoryS …
    -/
    intro A
    simp only [(splitting K).ι_desc_assoc, Obj.map_on_summand'_assoc K _ θ, (splitting K).ι_desc,
      assoc, Obj.map_on_summand' K' _ θ]
    /-
      case h
      C : Type u_1
      inst✝² : CategoryTheory.Category.{?u.79960, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      K✝ K'✝ : ChainComplex C Nat
      f✝ : Quiver.Hom K✝ K'✝
      Δ✝ Δ'✝ Δ'' : SimplexCategory
      inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
      K K' : ChainComplex C Nat
      f : Quiver.Hom K K'
      Δ' Δ : Opposite SimplexCategory
      θ : Quiver.Hom Δ' Δ
      A : SimplicialObject.Splitting.IndexSet Δ'
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicTopology.DoldKan.Γ₀.Obj.Ter …
    -/
    apply Obj.Termwise.mapMono_naturality_assoc
    /-
      🎉 no goals
    -/


/-- The functor `Γ₀' : ChainComplex C ℕ ⥤ SimplicialObject.Split C`
that induces `Γ₀ : ChainComplex C ℕ ⥤ SimplicialObject C`, which
shall be the inverse functor of the Dold-Kan equivalence for
abelian or pseudo-abelian categories. -/
@[simps]
def Γ₀' : ChainComplex C ℕ ⥤ SimplicialObject.Split C where
  obj K := SimplicialObject.Split.mk' (Γ₀.splitting K)
  map {K K'} f :=
    { F := Γ₀.map f
      f := f.f
      comm := fun n => by
        /-
          C : Type u_1
          inst✝² : CategoryTheory.Category.{?u.84336, u_1} C
          inst✝¹ : CategoryTheory.Preadditive C
          K✝ K'✝ : ChainComplex C Nat
          f✝ : Quiver.Hom K✝ K'✝
          Δ Δ' Δ'' : SimplexCategory
          inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
          K K' : ChainComplex C Nat
          f : Quiver.Hom K K'
          n : Nat
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (((fun K => SimplicialObject.Split.mk …
        -/
        dsimp
        /-
          C : Type u_1
          inst✝² : CategoryTheory.Category.{?u.84336, u_1} C
          inst✝¹ : CategoryTheory.Preadditive C
          K✝ K'✝ : ChainComplex C Nat
          f✝ : Quiver.Hom K✝ K'✝
          Δ Δ' Δ'' : SimplexCategory
          inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
          K K' : ChainComplex C Nat
          f : Quiver.Hom K K'
          n : Nat
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((AlgebraicTopology.DoldKan.Γ₀.splitt …
        -/
        simp only [← Splitting.cofan_inj_id, (Γ₀.splitting K).ι_desc]
        /-
          C : Type u_1
          inst✝² : CategoryTheory.Category.{?u.84336, u_1} C
          inst✝¹ : CategoryTheory.Preadditive C
          K✝ K'✝ : ChainComplex C Nat
          f✝ : Quiver.Hom K✝ K'✝
          Δ Δ' Δ'' : SimplexCategory
          inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
          K K' : ChainComplex C Nat
          f : Quiver.Hom K K'
          n : Nat
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (f.f (Opposite.unop (SimplicialObject …
        -/
        rfl }
        /-
          🎉 no goals
        -/


/-- The functor `Γ₀ : ChainComplex C ℕ ⥤ SimplicialObject C`, which is
the inverse functor of the Dold-Kan equivalence when `C` is an abelian
category, or more generally a pseudoabelian category. -/
@[simps!]
def Γ₀ : ChainComplex C ℕ ⥤ SimplicialObject C :=
  Γ₀' ⋙ Split.forget _


/-- The extension of `Γ₀ : ChainComplex C ℕ ⥤ SimplicialObject C`
on the idempotent completions. It shall be an equivalence of categories
for any additive category `C`. -/
@[simps!]
def Γ₂ : Karoubi (ChainComplex C ℕ) ⥤ Karoubi (SimplicialObject C) :=
  (CategoryTheory.Idempotents.functorExtension₂ _ _).obj Γ₀


theorem HigherFacesVanish.on_Γ₀_summand_id (K : ChainComplex C ℕ) (n : ℕ) :
    @HigherFacesVanish C _ _ (Γ₀.obj K) _ n (n + 1)
      (((Γ₀.splitting K).cofan _).inj (Splitting.IndexSet.id (op [n + 1]))) := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
    K : ChainComplex C Nat
    n : Nat
    ⊢ AlgebraicTopology.DoldKan.HigherFacesVanish (HAdd.hAdd n 1) (((AlgebraicTopo …
  -/
  intro j _
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
    K : ChainComplex C Nat
    n : Nat
    j : Fin (HAdd.hAdd n 1)
    a✝ : LE.le (HAdd.hAdd n 1) (HAdd.hAdd (↑j) (HAdd.hAdd n 1))
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (((AlgebraicTopology.DoldKan.Γ₀.split …
  -/
  have eq := Γ₀.Obj.mapMono_on_summand_id K (SimplexCategory.δ j.succ)
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
    K : ChainComplex C Nat
    n : Nat
    j : Fin (HAdd.hAdd n 1)
    a✝ : LE.le (HAdd.hAdd n 1) (HAdd.hAdd (↑j) (HAdd.hAdd n 1))
    eq : Eq (CategoryTheory.CategoryStruct.comp (((AlgebraicTopology.DoldKan.Γ₀.sp …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (((AlgebraicTopology.DoldKan.Γ₀.split …
  -/
  rw [Γ₀.Obj.Termwise.mapMono_eq_zero K, zero_comp] at eq; rotate_left
    /-
      case h₁
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_2, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
      K : ChainComplex C Nat
      n : Nat
      j : Fin (HAdd.hAdd n 1)
      a✝ : LE.le (HAdd.hAdd n 1) (HAdd.hAdd (↑j) (HAdd.hAdd n 1))
      eq : Eq (CategoryTheory.CategoryStruct.comp (((AlgebraicTopology.DoldKan.Γ₀.sp …
      ⊢ Ne (SimplexCategory.mk (HAdd.hAdd n 1)) (SimplexCategory.mk n)
    -/
  · intro h
    /-
      case h₁
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_2, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
      K : ChainComplex C Nat
      n : Nat
      j : Fin (HAdd.hAdd n 1)
      a✝ : LE.le (HAdd.hAdd n 1) (HAdd.hAdd (↑j) (HAdd.hAdd n 1))
      eq : Eq (CategoryTheory.CategoryStruct.comp (((AlgebraicTopology.DoldKan.Γ₀.sp …
      h : Eq (SimplexCategory.mk (HAdd.hAdd n 1)) (SimplexCategory.mk n)
      ⊢ False
    -/
    exact (Nat.succ_ne_self n) (congr_arg SimplexCategory.len h)
    /-
      🎉 no goals
    -/
    /-
      case h₂
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_2, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
      K : ChainComplex C Nat
      n : Nat
      j : Fin (HAdd.hAdd n 1)
      a✝ : LE.le (HAdd.hAdd n 1) (HAdd.hAdd (↑j) (HAdd.hAdd n 1))
      eq : Eq (CategoryTheory.CategoryStruct.comp (((AlgebraicTopology.DoldKan.Γ₀.sp …
      ⊢ Not (AlgebraicTopology.DoldKan.Isδ₀ (SimplexCategory.δ j.succ))
    -/
  · exact fun h => Fin.succ_ne_zero j (by simpa only [Isδ₀.iff] using h)
    /-
      🎉 no goals
    -/
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
    K : ChainComplex C Nat
    n : Nat
    j : Fin (HAdd.hAdd n 1)
    a✝ : LE.le (HAdd.hAdd n 1) (HAdd.hAdd (↑j) (HAdd.hAdd n 1))
    eq : Eq (CategoryTheory.CategoryStruct.comp (((AlgebraicTopology.DoldKan.Γ₀.sp …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (((AlgebraicTopology.DoldKan.Γ₀.split …
  -/
  exact eq
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
theorem PInfty_on_Γ₀_splitting_summand_eq_self (K : ChainComplex C ℕ) {n : ℕ} :
    ((Γ₀.splitting K).cofan _).inj (Splitting.IndexSet.id (op [n])) ≫
      (PInfty : K[Γ₀.obj K] ⟶ _).f n =
      ((Γ₀.splitting K).cofan _).inj (Splitting.IndexSet.id (op [n])) := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
    K : ChainComplex C Nat
    n : Nat
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (((AlgebraicTopology.DoldKan.Γ₀.split …
  -/
  rw [PInfty_f]
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
    K : ChainComplex C Nat
    n : Nat
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (((AlgebraicTopology.DoldKan.Γ₀.split …
  -/
  rcases n with _|n
    /-
      case zero
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_2, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
      K : ChainComplex C Nat
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (((AlgebraicTopology.DoldKan.Γ₀.split …
    -/
  · simpa only [P_f_0_eq] using comp_id _
    /-
      🎉 no goals
    -/
    /-
      case succ
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_2, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
      K : ChainComplex C Nat
      n : Nat
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (((AlgebraicTopology.DoldKan.Γ₀.split …
    -/
  · exact (HigherFacesVanish.on_Γ₀_summand_id K n).comp_P_eq_self
    /-
      🎉 no goals
    -/


