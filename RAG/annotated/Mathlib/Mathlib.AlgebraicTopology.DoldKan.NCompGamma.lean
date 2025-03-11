theorem PInfty_comp_map_mono_eq_zero (X : SimplicialObject C) {n : ℕ} {Δ' : SimplexCategory}
    (i : Δ' ⟶ [n]) [hi : Mono i] (h₁ : Δ'.len ≠ n) (h₂ : ¬Isδ₀ i) :
    PInfty.f n ≫ X.map i.op = 0 := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    X : CategoryTheory.SimplicialObject C
    n : Nat
    Δ' : SimplexCategory
    i : Quiver.Hom Δ' (SimplexCategory.mk n)
    hi : CategoryTheory.Mono i
    h₁ : Ne Δ'.len n
    h₂ : Not (AlgebraicTopology.DoldKan.Isδ₀ i)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicTopology.DoldKan.PInfty.f n …
  -/
  induction' Δ' using SimplexCategory.rec with m
  obtain ⟨k, hk⟩ := Nat.exists_eq_add_of_lt (len_lt_of_mono i fun h => by
        rw [← h] at h₁
        exact h₁ rfl)
  /-
    case h.intro
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    X : CategoryTheory.SimplicialObject C
    n m : Nat
    i : Quiver.Hom (SimplexCategory.mk m) (SimplexCategory.mk n)
    hi : CategoryTheory.Mono i
    h₁ : Ne (SimplexCategory.mk m).len n
    h₂ : Not (AlgebraicTopology.DoldKan.Isδ₀ i)
    k : Nat
    hk : Eq (SimplexCategory.mk n).len (HAdd.hAdd (HAdd.hAdd (SimplexCategory.mk m …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicTopology.DoldKan.PInfty.f n …
  -/
  simp only [len_mk] at hk
  /-
    case h.intro
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    X : CategoryTheory.SimplicialObject C
    n m : Nat
    i : Quiver.Hom (SimplexCategory.mk m) (SimplexCategory.mk n)
    hi : CategoryTheory.Mono i
    h₁ : Ne (SimplexCategory.mk m).len n
    h₂ : Not (AlgebraicTopology.DoldKan.Isδ₀ i)
    k : Nat
    hk : Eq n (HAdd.hAdd (HAdd.hAdd m k) 1)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicTopology.DoldKan.PInfty.f n …
  -/
  rcases k with _|k
    /-
      case h.intro.zero
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      X : CategoryTheory.SimplicialObject C
      n m : Nat
      i : Quiver.Hom (SimplexCategory.mk m) (SimplexCategory.mk n)
      hi : CategoryTheory.Mono i
      h₁ : Ne (SimplexCategory.mk m).len n
      h₂ : Not (AlgebraicTopology.DoldKan.Isδ₀ i)
      hk : Eq n (HAdd.hAdd (HAdd.hAdd m 0) 1)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicTopology.DoldKan.PInfty.f n …
    -/
  · change n = m + 1 at hk
    /-
      case h.intro.zero
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      X : CategoryTheory.SimplicialObject C
      n m : Nat
      i : Quiver.Hom (SimplexCategory.mk m) (SimplexCategory.mk n)
      hi : CategoryTheory.Mono i
      h₁ : Ne (SimplexCategory.mk m).len n
      h₂ : Not (AlgebraicTopology.DoldKan.Isδ₀ i)
      hk : Eq n (HAdd.hAdd m 1)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicTopology.DoldKan.PInfty.f n …
    -/
    subst hk
    /-
      case h.intro.zero
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      X : CategoryTheory.SimplicialObject C
      m : Nat
      i : Quiver.Hom (SimplexCategory.mk m) (SimplexCategory.mk (HAdd.hAdd m 1))
      hi : CategoryTheory.Mono i
      h₁ : Ne (SimplexCategory.mk m).len (HAdd.hAdd m 1)
      h₂ : Not (AlgebraicTopology.DoldKan.Isδ₀ i)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicTopology.DoldKan.PInfty.f ( …
    -/
    obtain ⟨j, rfl⟩ := eq_δ_of_mono i
    /-
      case h.intro.zero.intro
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      X : CategoryTheory.SimplicialObject C
      m : Nat
      h₁ : Ne (SimplexCategory.mk m).len (HAdd.hAdd m 1)
      j : Fin (HAdd.hAdd m 2)
      hi : CategoryTheory.Mono (SimplexCategory.δ j)
      h₂ : Not (AlgebraicTopology.DoldKan.Isδ₀ (SimplexCategory.δ j))
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicTopology.DoldKan.PInfty.f ( …
    -/
    rw [Isδ₀.iff] at h₂
    have h₃ : 1 ≤ (j : ℕ) := by
      by_contra h
      exact h₂ (by simpa only [Fin.ext_iff, not_le, Nat.lt_one_iff] using h)
    /-
      case h.intro.zero.intro
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      X : CategoryTheory.SimplicialObject C
      m : Nat
      h₁ : Ne (SimplexCategory.mk m).len (HAdd.hAdd m 1)
      j : Fin (HAdd.hAdd m 2)
      hi : CategoryTheory.Mono (SimplexCategory.δ j)
      h₂ : Not (Eq j 0)
      h₃ : LE.le 1 ↑j
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicTopology.DoldKan.PInfty.f ( …
    -/
    exact (HigherFacesVanish.of_P (m + 1) m).comp_δ_eq_zero j h₂ (by omega)
    /-
      🎉 no goals
    -/
    /-
      case h.intro.succ
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      X : CategoryTheory.SimplicialObject C
      n m : Nat
      i : Quiver.Hom (SimplexCategory.mk m) (SimplexCategory.mk n)
      hi : CategoryTheory.Mono i
      h₁ : Ne (SimplexCategory.mk m).len n
      h₂ : Not (AlgebraicTopology.DoldKan.Isδ₀ i)
      k : Nat
      hk : Eq n (HAdd.hAdd (HAdd.hAdd m (HAdd.hAdd k 1)) 1)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicTopology.DoldKan.PInfty.f n …
    -/
  · simp only [Nat.succ_eq_add_one, ← add_assoc] at hk
    /-
      case h.intro.succ
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      X : CategoryTheory.SimplicialObject C
      n m : Nat
      i : Quiver.Hom (SimplexCategory.mk m) (SimplexCategory.mk n)
      hi : CategoryTheory.Mono i
      h₁ : Ne (SimplexCategory.mk m).len n
      h₂ : Not (AlgebraicTopology.DoldKan.Isδ₀ i)
      k : Nat
      hk : Eq n (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd m k) 1) 1)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicTopology.DoldKan.PInfty.f n …
    -/
    clear h₂ hi
    /-
      case h.intro.succ
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      X : CategoryTheory.SimplicialObject C
      n m : Nat
      i : Quiver.Hom (SimplexCategory.mk m) (SimplexCategory.mk n)
      h₁ : Ne (SimplexCategory.mk m).len n
      k : Nat
      hk : Eq n (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd m k) 1) 1)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicTopology.DoldKan.PInfty.f n …
    -/
    subst hk
    obtain ⟨j₁ : Fin (_ + 1), i, rfl⟩ :=
      eq_comp_δ_of_not_surjective i fun h => by
        have h' := len_le_of_epi (SimplexCategory.epi_iff_surjective.2 h)
        dsimp at h'
        omega
    obtain ⟨j₂, i, rfl⟩ :=
      eq_comp_δ_of_not_surjective i fun h => by
        have h' := len_le_of_epi (SimplexCategory.epi_iff_surjective.2 h)
        dsimp at h'
        omega
    /-
      case h.intro.succ.intro.intro.intro.intro
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      X : CategoryTheory.SimplicialObject C
      m k : Nat
      h₁ : Ne (SimplexCategory.mk m).len (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd m k) 1) 1)
      j₁ : Fin (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd m k) 2) 1)
      j₂ : Fin (HAdd.hAdd (HAdd.hAdd m k) 2)
      i : Quiver.Hom (SimplexCategory.mk m) (SimplexCategory.mk (HAdd.hAdd m k))
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicTopology.DoldKan.PInfty.f ( …
    -/
    by_cases hj₁ : j₁ = 0
      /-
        case pos
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
        inst✝ : CategoryTheory.Preadditive C
        X : CategoryTheory.SimplicialObject C
        m k : Nat
        h₁ : Ne (SimplexCategory.mk m).len (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd m k) 1) 1)
        j₁ : Fin (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd m k) 2) 1)
        j₂ : Fin (HAdd.hAdd (HAdd.hAdd m k) 2)
        i : Quiver.Hom (SimplexCategory.mk m) (SimplexCategory.mk (HAdd.hAdd m k))
        hj₁ : Eq j₁ 0
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicTopology.DoldKan.PInfty.f ( …
      -/
    · subst hj₁
      /-
        case pos
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
        inst✝ : CategoryTheory.Preadditive C
        X : CategoryTheory.SimplicialObject C
        m k : Nat
        h₁ : Ne (SimplexCategory.mk m).len (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd m k) 1) 1)
        j₂ : Fin (HAdd.hAdd (HAdd.hAdd m k) 2)
        i : Quiver.Hom (SimplexCategory.mk m) (SimplexCategory.mk (HAdd.hAdd m k))
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicTopology.DoldKan.PInfty.f ( …
      -/
      rw [assoc, ← SimplexCategory.δ_comp_δ'' (Fin.zero_le _)]
      /-
        case pos
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
        inst✝ : CategoryTheory.Preadditive C
        X : CategoryTheory.SimplicialObject C
        m k : Nat
        h₁ : Ne (SimplexCategory.mk m).len (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd m k) 1) 1)
        j₂ : Fin (HAdd.hAdd (HAdd.hAdd m k) 2)
        i : Quiver.Hom (SimplexCategory.mk m) (SimplexCategory.mk (HAdd.hAdd m k))
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicTopology.DoldKan.PInfty.f ( …
      -/
      simp only [op_comp, X.map_comp, assoc, PInfty_f]
      /-
        case pos
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
        inst✝ : CategoryTheory.Preadditive C
        X : CategoryTheory.SimplicialObject C
        m k : Nat
        h₁ : Ne (SimplexCategory.mk m).len (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd m k) 1) 1)
        j₂ : Fin (HAdd.hAdd (HAdd.hAdd m k) 2)
        i : Quiver.Hom (SimplexCategory.mk m) (SimplexCategory.mk (HAdd.hAdd m k))
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((AlgebraicTopology.DoldKan.P (HAdd.h …
      -/
      erw [(HigherFacesVanish.of_P _ _).comp_δ_eq_zero_assoc _ j₂.succ_ne_zero, zero_comp]
      /-
        case pos.hj₂
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
        inst✝ : CategoryTheory.Preadditive C
        X : CategoryTheory.SimplicialObject C
        m k : Nat
        h₁ : Ne (SimplexCategory.mk m).len (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd m k) 1) 1)
        j₂ : Fin (HAdd.hAdd (HAdd.hAdd m k) 2)
        i : Quiver.Hom (SimplexCategory.mk m) (SimplexCategory.mk (HAdd.hAdd m k))
        ⊢ LE.le (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd m k) 1) 2) (HAdd.hAdd (↑j₂.succ) (HAd …
      -/
      simp only [Nat.succ_eq_add_one, Nat.add, Fin.succ]
      /-
        case pos.hj₂
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
        inst✝ : CategoryTheory.Preadditive C
        X : CategoryTheory.SimplicialObject C
        m k : Nat
        h₁ : Ne (SimplexCategory.mk m).len (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd m k) 1) 1)
        j₂ : Fin (HAdd.hAdd (HAdd.hAdd m k) 2)
        i : Quiver.Hom (SimplexCategory.mk m) (SimplexCategory.mk (HAdd.hAdd m k))
        ⊢ LE.le (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd m k) 1) 2) (HAdd.hAdd (HAdd.hAdd (↑j₂ …
      -/
      omega
      /-
        🎉 no goals
      -/
      /-
        case neg
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
        inst✝ : CategoryTheory.Preadditive C
        X : CategoryTheory.SimplicialObject C
        m k : Nat
        h₁ : Ne (SimplexCategory.mk m).len (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd m k) 1) 1)
        j₁ : Fin (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd m k) 2) 1)
        j₂ : Fin (HAdd.hAdd (HAdd.hAdd m k) 2)
        i : Quiver.Hom (SimplexCategory.mk m) (SimplexCategory.mk (HAdd.hAdd m k))
        hj₁ : Not (Eq j₁ 0)
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicTopology.DoldKan.PInfty.f ( …
      -/
    · simp only [op_comp, X.map_comp, assoc, PInfty_f]
      /-
        case neg
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
        inst✝ : CategoryTheory.Preadditive C
        X : CategoryTheory.SimplicialObject C
        m k : Nat
        h₁ : Ne (SimplexCategory.mk m).len (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd m k) 1) 1)
        j₁ : Fin (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd m k) 2) 1)
        j₂ : Fin (HAdd.hAdd (HAdd.hAdd m k) 2)
        i : Quiver.Hom (SimplexCategory.mk m) (SimplexCategory.mk (HAdd.hAdd m k))
        hj₁ : Not (Eq j₁ 0)
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((AlgebraicTopology.DoldKan.P (HAdd.h …
      -/
      erw [(HigherFacesVanish.of_P _ _).comp_δ_eq_zero_assoc _ hj₁, zero_comp]
      /-
        case neg.hj₂
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
        inst✝ : CategoryTheory.Preadditive C
        X : CategoryTheory.SimplicialObject C
        m k : Nat
        h₁ : Ne (SimplexCategory.mk m).len (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd m k) 1) 1)
        j₁ : Fin (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd m k) 2) 1)
        j₂ : Fin (HAdd.hAdd (HAdd.hAdd m k) 2)
        i : Quiver.Hom (SimplexCategory.mk m) (SimplexCategory.mk (HAdd.hAdd m k))
        hj₁ : Not (Eq j₁ 0)
        ⊢ LE.le (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd m k) 1) 2) (HAdd.hAdd (↑j₁) (HAdd.hAd …
      -/
      by_contra
      /-
        case neg.hj₂
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
        inst✝ : CategoryTheory.Preadditive C
        X : CategoryTheory.SimplicialObject C
        m k : Nat
        h₁ : Ne (SimplexCategory.mk m).len (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd m k) 1) 1)
        j₁ : Fin (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd m k) 2) 1)
        j₂ : Fin (HAdd.hAdd (HAdd.hAdd m k) 2)
        i : Quiver.Hom (SimplexCategory.mk m) (SimplexCategory.mk (HAdd.hAdd m k))
        hj₁ : Not (Eq j₁ 0)
        x✝ : Not (LE.le (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd m k) 1) 2) (HAdd.hAdd (↑j₁) ( …
        ⊢ False
      -/
      exact hj₁ (by simp only [Fin.ext_iff, Fin.val_zero]; omega)
      /-
        🎉 no goals
      -/


@[reassoc]
theorem Γ₀_obj_termwise_mapMono_comp_PInfty (X : SimplicialObject C) {Δ Δ' : SimplexCategory}
    (i : Δ ⟶ Δ') [Mono i] :
    Γ₀.Obj.Termwise.mapMono (AlternatingFaceMapComplex.obj X) i ≫ PInfty.f Δ.len =
      PInfty.f Δ'.len ≫ X.map i.op := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    X : CategoryTheory.SimplicialObject C
    Δ Δ' : SimplexCategory
    i : Quiver.Hom Δ Δ'
    inst✝ : CategoryTheory.Mono i
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicTopology.DoldKan.Γ₀.Obj.Ter …
  -/
  induction' Δ using SimplexCategory.rec with n
  /-
    case h
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    X : CategoryTheory.SimplicialObject C
    Δ' : SimplexCategory
    n : Nat
    i : Quiver.Hom (SimplexCategory.mk n) Δ'
    inst✝ : CategoryTheory.Mono i
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicTopology.DoldKan.Γ₀.Obj.Ter …
  -/
  induction' Δ' using SimplexCategory.rec with n'
  /-
    case h.h
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    X : CategoryTheory.SimplicialObject C
    n n' : Nat
    i : Quiver.Hom (SimplexCategory.mk n) (SimplexCategory.mk n')
    inst✝ : CategoryTheory.Mono i
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicTopology.DoldKan.Γ₀.Obj.Ter …
  -/
  dsimp
  -- We start with the case `i` is an identity
  /-
    case h.h
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    X : CategoryTheory.SimplicialObject C
    n n' : Nat
    i : Quiver.Hom (SimplexCategory.mk n) (SimplexCategory.mk n')
    inst✝ : CategoryTheory.Mono i
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicTopology.DoldKan.Γ₀.Obj.Ter …
  -/
  by_cases h : n = n'
    /-
      case pos
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_2, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      X : CategoryTheory.SimplicialObject C
      n n' : Nat
      i : Quiver.Hom (SimplexCategory.mk n) (SimplexCategory.mk n')
      inst✝ : CategoryTheory.Mono i
      h : Eq n n'
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicTopology.DoldKan.Γ₀.Obj.Ter …
    -/
  · subst h
    /-
      case pos
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_2, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      X : CategoryTheory.SimplicialObject C
      n : Nat
      i : Quiver.Hom (SimplexCategory.mk n) (SimplexCategory.mk n)
      inst✝ : CategoryTheory.Mono i
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicTopology.DoldKan.Γ₀.Obj.Ter …
    -/
    simp only [SimplexCategory.eq_id_of_mono i, Γ₀.Obj.Termwise.mapMono_id, op_id, X.map_id]
    /-
      case pos
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_2, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      X : CategoryTheory.SimplicialObject C
      n : Nat
      i : Quiver.Hom (SimplexCategory.mk n) (SimplexCategory.mk n)
      inst✝ : CategoryTheory.Mono i
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id ((A …
    -/
    dsimp
    /-
      case pos
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_2, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      X : CategoryTheory.SimplicialObject C
      n : Nat
      i : Quiver.Hom (SimplexCategory.mk n) (SimplexCategory.mk n)
      inst✝ : CategoryTheory.Mono i
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id (X. …
    -/
    simp only [id_comp, comp_id]
    /-
      🎉 no goals
    -/
  /-
    case neg
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    X : CategoryTheory.SimplicialObject C
    n n' : Nat
    i : Quiver.Hom (SimplexCategory.mk n) (SimplexCategory.mk n')
    inst✝ : CategoryTheory.Mono i
    h : Not (Eq n n')
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicTopology.DoldKan.Γ₀.Obj.Ter …
  -/
  by_cases hi : Isδ₀ i
  -- The case `i = δ 0`
    /-
      case pos
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_2, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      X : CategoryTheory.SimplicialObject C
      n n' : Nat
      i : Quiver.Hom (SimplexCategory.mk n) (SimplexCategory.mk n')
      inst✝ : CategoryTheory.Mono i
      h : Not (Eq n n')
      hi : AlgebraicTopology.DoldKan.Isδ₀ i
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicTopology.DoldKan.Γ₀.Obj.Ter …
    -/
  · have h' : n' = n + 1 := hi.left
    /-
      case pos
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_2, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      X : CategoryTheory.SimplicialObject C
      n n' : Nat
      i : Quiver.Hom (SimplexCategory.mk n) (SimplexCategory.mk n')
      inst✝ : CategoryTheory.Mono i
      h : Not (Eq n n')
      hi : AlgebraicTopology.DoldKan.Isδ₀ i
      h' : Eq n' (HAdd.hAdd n 1)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicTopology.DoldKan.Γ₀.Obj.Ter …
    -/
    subst h'
    /-
      case pos
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_2, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      X : CategoryTheory.SimplicialObject C
      n : Nat
      i : Quiver.Hom (SimplexCategory.mk n) (SimplexCategory.mk (HAdd.hAdd n 1))
      inst✝ : CategoryTheory.Mono i
      h : Not (Eq n (HAdd.hAdd n 1))
      hi : AlgebraicTopology.DoldKan.Isδ₀ i
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicTopology.DoldKan.Γ₀.Obj.Ter …
    -/
    simp only [Γ₀.Obj.Termwise.mapMono_δ₀' _ i hi]
    /-
      case pos
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_2, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      X : CategoryTheory.SimplicialObject C
      n : Nat
      i : Quiver.Hom (SimplexCategory.mk n) (SimplexCategory.mk (HAdd.hAdd n 1))
      inst✝ : CategoryTheory.Mono i
      h : Not (Eq n (HAdd.hAdd n 1))
      hi : AlgebraicTopology.DoldKan.Isδ₀ i
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((AlgebraicTopology.AlternatingFaceMa …
    -/
    dsimp
    /-
      case pos
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_2, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      X : CategoryTheory.SimplicialObject C
      n : Nat
      i : Quiver.Hom (SimplexCategory.mk n) (SimplexCategory.mk (HAdd.hAdd n 1))
      inst✝ : CategoryTheory.Mono i
      h : Not (Eq n (HAdd.hAdd n 1))
      hi : AlgebraicTopology.DoldKan.Isδ₀ i
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((AlgebraicTopology.AlternatingFaceMa …
    -/
    rw [← PInfty.comm _ n, AlternatingFaceMapComplex.obj_d_eq]
    /-
      case pos
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_2, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      X : CategoryTheory.SimplicialObject C
      n : Nat
      i : Quiver.Hom (SimplexCategory.mk n) (SimplexCategory.mk (HAdd.hAdd n 1))
      inst✝ : CategoryTheory.Mono i
      h : Not (Eq n (HAdd.hAdd n 1))
      hi : AlgebraicTopology.DoldKan.Isδ₀ i
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicTopology.DoldKan.PInfty.f ( …
    -/
    simp only [eq_self_iff_true, id_comp, if_true, Preadditive.comp_sum]
    /-
      case pos
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_2, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      X : CategoryTheory.SimplicialObject C
      n : Nat
      i : Quiver.Hom (SimplexCategory.mk n) (SimplexCategory.mk (HAdd.hAdd n 1))
      inst✝ : CategoryTheory.Mono i
      h : Not (Eq n (HAdd.hAdd n 1))
      hi : AlgebraicTopology.DoldKan.Isδ₀ i
      ⊢ Eq (Finset.univ.sum fun j => CategoryTheory.CategoryStruct.comp (AlgebraicTo …
    -/
    rw [Finset.sum_eq_single (0 : Fin (n + 2))]
    /-
      case pos
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_2, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      X : CategoryTheory.SimplicialObject C
      n : Nat
      i : Quiver.Hom (SimplexCategory.mk n) (SimplexCategory.mk (HAdd.hAdd n 1))
      inst✝ : CategoryTheory.Mono i
      h : Not (Eq n (HAdd.hAdd n 1))
      hi : AlgebraicTopology.DoldKan.Isδ₀ i
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicTopology.DoldKan.PInfty.f ( …
    -/
    rotate_left
      /-
        case pos.h₀
        C : Type u_1
        inst✝² : CategoryTheory.Category.{u_2, u_1} C
        inst✝¹ : CategoryTheory.Preadditive C
        X : CategoryTheory.SimplicialObject C
        n : Nat
        i : Quiver.Hom (SimplexCategory.mk n) (SimplexCategory.mk (HAdd.hAdd n 1))
        inst✝ : CategoryTheory.Mono i
        h : Not (Eq n (HAdd.hAdd n 1))
        hi : AlgebraicTopology.DoldKan.Isδ₀ i
        ⊢ ∀ (b : Fin (HAdd.hAdd n 2)), Membership.mem Finset.univ b → Ne b 0 → Eq (Cat …
      -/
    · intro b _ hb
      /-
        case pos.h₀
        C : Type u_1
        inst✝² : CategoryTheory.Category.{u_2, u_1} C
        inst✝¹ : CategoryTheory.Preadditive C
        X : CategoryTheory.SimplicialObject C
        n : Nat
        i : Quiver.Hom (SimplexCategory.mk n) (SimplexCategory.mk (HAdd.hAdd n 1))
        inst✝ : CategoryTheory.Mono i
        h : Not (Eq n (HAdd.hAdd n 1))
        hi : AlgebraicTopology.DoldKan.Isδ₀ i
        b : Fin (HAdd.hAdd n 2)
        a✝ : Membership.mem Finset.univ b
        hb : Ne b 0
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicTopology.DoldKan.PInfty.f ( …
      -/
      rw [Preadditive.comp_zsmul]
      erw [PInfty_comp_map_mono_eq_zero X (SimplexCategory.δ b) h
          (by
            rw [Isδ₀.iff]
            exact hb),
        zsmul_zero]
      /-
        case pos.h₁
        C : Type u_1
        inst✝² : CategoryTheory.Category.{u_2, u_1} C
        inst✝¹ : CategoryTheory.Preadditive C
        X : CategoryTheory.SimplicialObject C
        n : Nat
        i : Quiver.Hom (SimplexCategory.mk n) (SimplexCategory.mk (HAdd.hAdd n 1))
        inst✝ : CategoryTheory.Mono i
        h : Not (Eq n (HAdd.hAdd n 1))
        hi : AlgebraicTopology.DoldKan.Isδ₀ i
        ⊢ Not (Membership.mem Finset.univ 0) → Eq (CategoryTheory.CategoryStruct.comp  …
      -/
    · simp only [Finset.mem_univ, not_true, IsEmpty.forall_iff]
      /-
        🎉 no goals
      -/
      /-
        case pos
        C : Type u_1
        inst✝² : CategoryTheory.Category.{u_2, u_1} C
        inst✝¹ : CategoryTheory.Preadditive C
        X : CategoryTheory.SimplicialObject C
        n : Nat
        i : Quiver.Hom (SimplexCategory.mk n) (SimplexCategory.mk (HAdd.hAdd n 1))
        inst✝ : CategoryTheory.Mono i
        h : Not (Eq n (HAdd.hAdd n 1))
        hi : AlgebraicTopology.DoldKan.Isδ₀ i
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicTopology.DoldKan.PInfty.f ( …
      -/
    · simp only [hi.eq_δ₀, Fin.val_zero, pow_zero, one_zsmul]
      /-
        case pos
        C : Type u_1
        inst✝² : CategoryTheory.Category.{u_2, u_1} C
        inst✝¹ : CategoryTheory.Preadditive C
        X : CategoryTheory.SimplicialObject C
        n : Nat
        i : Quiver.Hom (SimplexCategory.mk n) (SimplexCategory.mk (HAdd.hAdd n 1))
        inst✝ : CategoryTheory.Mono i
        h : Not (Eq n (HAdd.hAdd n 1))
        hi : AlgebraicTopology.DoldKan.Isδ₀ i
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicTopology.DoldKan.PInfty.f ( …
      -/
      rfl
      /-
        🎉 no goals
      -/
  -- The case `i ≠ δ 0`
    /-
      case neg
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_2, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      X : CategoryTheory.SimplicialObject C
      n n' : Nat
      i : Quiver.Hom (SimplexCategory.mk n) (SimplexCategory.mk n')
      inst✝ : CategoryTheory.Mono i
      h : Not (Eq n n')
      hi : Not (AlgebraicTopology.DoldKan.Isδ₀ i)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicTopology.DoldKan.Γ₀.Obj.Ter …
    -/
  · rw [Γ₀.Obj.Termwise.mapMono_eq_zero _ i _ hi, zero_comp]
    /-
      case neg
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_2, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      X : CategoryTheory.SimplicialObject C
      n n' : Nat
      i : Quiver.Hom (SimplexCategory.mk n) (SimplexCategory.mk n')
      inst✝ : CategoryTheory.Mono i
      h : Not (Eq n n')
      hi : Not (AlgebraicTopology.DoldKan.Isδ₀ i)
      ⊢ Eq 0 (CategoryTheory.CategoryStruct.comp (AlgebraicTopology.DoldKan.PInfty.f …
    -/
    swap
      /-
        C : Type u_1
        inst✝² : CategoryTheory.Category.{u_2, u_1} C
        inst✝¹ : CategoryTheory.Preadditive C
        X : CategoryTheory.SimplicialObject C
        n n' : Nat
        i : Quiver.Hom (SimplexCategory.mk n) (SimplexCategory.mk n')
        inst✝ : CategoryTheory.Mono i
        h : Not (Eq n n')
        hi : Not (AlgebraicTopology.DoldKan.Isδ₀ i)
        ⊢ Ne (SimplexCategory.mk n') (SimplexCategory.mk n)
      -/
    · by_contra h'
      /-
        C : Type u_1
        inst✝² : CategoryTheory.Category.{u_2, u_1} C
        inst✝¹ : CategoryTheory.Preadditive C
        X : CategoryTheory.SimplicialObject C
        n n' : Nat
        i : Quiver.Hom (SimplexCategory.mk n) (SimplexCategory.mk n')
        inst✝ : CategoryTheory.Mono i
        h : Not (Eq n n')
        hi : Not (AlgebraicTopology.DoldKan.Isδ₀ i)
        h' : Eq (SimplexCategory.mk n') (SimplexCategory.mk n)
        ⊢ False
      -/
      exact h (congr_arg SimplexCategory.len h'.symm)
      /-
        🎉 no goals
      -/
    /-
      case neg
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_2, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      X : CategoryTheory.SimplicialObject C
      n n' : Nat
      i : Quiver.Hom (SimplexCategory.mk n) (SimplexCategory.mk n')
      inst✝ : CategoryTheory.Mono i
      h : Not (Eq n n')
      hi : Not (AlgebraicTopology.DoldKan.Isδ₀ i)
      ⊢ Eq 0 (CategoryTheory.CategoryStruct.comp (AlgebraicTopology.DoldKan.PInfty.f …
    -/
    rw [PInfty_comp_map_mono_eq_zero]
      /-
        case neg.h₁
        C : Type u_1
        inst✝² : CategoryTheory.Category.{u_2, u_1} C
        inst✝¹ : CategoryTheory.Preadditive C
        X : CategoryTheory.SimplicialObject C
        n n' : Nat
        i : Quiver.Hom (SimplexCategory.mk n) (SimplexCategory.mk n')
        inst✝ : CategoryTheory.Mono i
        h : Not (Eq n n')
        hi : Not (AlgebraicTopology.DoldKan.Isδ₀ i)
        ⊢ Ne (SimplexCategory.mk n).len n'
      -/
    · exact h
      /-
        🎉 no goals
      -/
      /-
        case neg.h₂
        C : Type u_1
        inst✝² : CategoryTheory.Category.{u_2, u_1} C
        inst✝¹ : CategoryTheory.Preadditive C
        X : CategoryTheory.SimplicialObject C
        n n' : Nat
        i : Quiver.Hom (SimplexCategory.mk n) (SimplexCategory.mk n')
        inst✝ : CategoryTheory.Mono i
        h : Not (Eq n n')
        hi : Not (AlgebraicTopology.DoldKan.Isδ₀ i)
        ⊢ Not (AlgebraicTopology.DoldKan.Isδ₀ i)
      -/
    · by_contra h'
      /-
        case neg.h₂
        C : Type u_1
        inst✝² : CategoryTheory.Category.{u_2, u_1} C
        inst✝¹ : CategoryTheory.Preadditive C
        X : CategoryTheory.SimplicialObject C
        n n' : Nat
        i : Quiver.Hom (SimplexCategory.mk n) (SimplexCategory.mk n')
        inst✝ : CategoryTheory.Mono i
        h : Not (Eq n n')
        hi : Not (AlgebraicTopology.DoldKan.Isδ₀ i)
        h' : AlgebraicTopology.DoldKan.Isδ₀ i
        ⊢ False
      -/
      exact hi h'
      /-
        🎉 no goals
      -/


/-- The natural transformation `N₁ ⋙ Γ₂ ⟶ toKaroubi (SimplicialObject C)`. -/
@[simps]
def natTrans : (N₁ : SimplicialObject C ⥤ _) ⋙ Γ₂ ⟶ toKaroubi _ where
  app X :=
    { f :=
        { app := fun Δ => (Γ₀.splitting K[X]).desc Δ fun A => PInfty.f A.1.unop.len ≫ X.map A.e.op
          naturality := fun Δ Δ' θ => by
            /-
              C : Type u_1
              inst✝² : CategoryTheory.Category.{?u.22112, u_1} C
              inst✝¹ : CategoryTheory.Preadditive C
              inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
              X : CategoryTheory.SimplicialObject C
              Δ Δ' : Opposite SimplexCategory
              θ : Quiver.Hom Δ Δ'
              ⊢ Eq (CategoryTheory.CategoryStruct.comp (((AlgebraicTopology.DoldKan.N₁.comp  …
            -/
            apply (Γ₀.splitting K[X]).hom_ext'
            /-
              case h
              C : Type u_1
              inst✝² : CategoryTheory.Category.{?u.22112, u_1} C
              inst✝¹ : CategoryTheory.Preadditive C
              inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
              X : CategoryTheory.SimplicialObject C
              Δ Δ' : Opposite SimplexCategory
              θ : Quiver.Hom Δ Δ'
              ⊢ ∀ (A : SimplicialObject.Splitting.IndexSet Δ), Eq (CategoryTheory.CategorySt …
            -/
            intro A
            /-
              case h
              C : Type u_1
              inst✝² : CategoryTheory.Category.{?u.22112, u_1} C
              inst✝¹ : CategoryTheory.Preadditive C
              inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
              X : CategoryTheory.SimplicialObject C
              Δ Δ' : Opposite SimplexCategory
              θ : Quiver.Hom Δ Δ'
              A : SimplicialObject.Splitting.IndexSet Δ
              ⊢ Eq (CategoryTheory.CategoryStruct.comp (((AlgebraicTopology.DoldKan.Γ₀.split …
            -/
            change _ ≫ (Γ₀.obj K[X]).map θ ≫ _ = _
            simp only [Splitting.ι_desc_assoc, assoc, Γ₀.Obj.map_on_summand'_assoc,
              Splitting.ι_desc]
            /-
              case h
              C : Type u_1
              inst✝² : CategoryTheory.Category.{?u.22112, u_1} C
              inst✝¹ : CategoryTheory.Preadditive C
              inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
              X : CategoryTheory.SimplicialObject C
              Δ Δ' : Opposite SimplexCategory
              θ : Quiver.Hom Δ Δ'
              A : SimplicialObject.Splitting.IndexSet Δ
              ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicTopology.DoldKan.Γ₀.Obj.Ter …
            -/
            erw [Γ₀_obj_termwise_mapMono_comp_PInfty_assoc X (image.ι (θ.unop ≫ A.e))]
            /-
              case h
              C : Type u_1
              inst✝² : CategoryTheory.Category.{?u.22112, u_1} C
              inst✝¹ : CategoryTheory.Preadditive C
              inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
              X : CategoryTheory.SimplicialObject C
              Δ Δ' : Opposite SimplexCategory
              θ : Quiver.Hom Δ Δ'
              A : SimplicialObject.Splitting.IndexSet Δ
              ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicTopology.DoldKan.PInfty.f ( …
            -/
            dsimp only [toKaroubi]
            /-
              case h
              C : Type u_1
              inst✝² : CategoryTheory.Category.{?u.22112, u_1} C
              inst✝¹ : CategoryTheory.Preadditive C
              inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
              X : CategoryTheory.SimplicialObject C
              Δ Δ' : Opposite SimplexCategory
              θ : Quiver.Hom Δ Δ'
              A : SimplicialObject.Splitting.IndexSet Δ
              ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicTopology.DoldKan.PInfty.f ( …
            -/
            simp only [← X.map_comp]
            /-
              case h
              C : Type u_1
              inst✝² : CategoryTheory.Category.{?u.22112, u_1} C
              inst✝¹ : CategoryTheory.Preadditive C
              inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
              X : CategoryTheory.SimplicialObject C
              Δ Δ' : Opposite SimplexCategory
              θ : Quiver.Hom Δ Δ'
              A : SimplicialObject.Splitting.IndexSet Δ
              ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicTopology.DoldKan.PInfty.f ( …
            -/
            congr 2
            /-
              case h.e_a.e_a
              C : Type u_1
              inst✝² : CategoryTheory.Category.{?u.22112, u_1} C
              inst✝¹ : CategoryTheory.Preadditive C
              inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
              X : CategoryTheory.SimplicialObject C
              Δ Δ' : Opposite SimplexCategory
              θ : Quiver.Hom Δ Δ'
              A : SimplicialObject.Splitting.IndexSet Δ
              ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.image.ι (Categ …
            -/
            simp only [eqToHom_refl, id_comp, comp_id, ← op_comp]
            /-
              case h.e_a.e_a
              C : Type u_1
              inst✝² : CategoryTheory.Category.{?u.22112, u_1} C
              inst✝¹ : CategoryTheory.Preadditive C
              inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
              X : CategoryTheory.SimplicialObject C
              Δ Δ' : Opposite SimplexCategory
              θ : Quiver.Hom Δ Δ'
              A : SimplicialObject.Splitting.IndexSet Δ
              ⊢ Eq (CategoryTheory.CategoryStruct.comp (A.pull θ).e (CategoryTheory.Limits.i …
            -/
            exact Quiver.Hom.unop_inj (A.fac_pull θ) }
            /-
              🎉 no goals
            -/
      comm := by
        /-
          C : Type u_1
          inst✝² : CategoryTheory.Category.{?u.22112, u_1} C
          inst✝¹ : CategoryTheory.Preadditive C
          inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
          X : CategoryTheory.SimplicialObject C
          ⊢ Eq { app := fun Δ => (AlgebraicTopology.DoldKan.Γ₀.splitting (AlgebraicTopol …
        -/
        apply (Γ₀.splitting K[X]).hom_ext
        /-
          case h
          C : Type u_1
          inst✝² : CategoryTheory.Category.{?u.22112, u_1} C
          inst✝¹ : CategoryTheory.Preadditive C
          inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
          X : CategoryTheory.SimplicialObject C
          ⊢ ∀ (n : Nat), Eq ((AlgebraicTopology.DoldKan.Γ₀.splitting (AlgebraicTopology. …
        -/
        intro n
        /-
          case h
          C : Type u_1
          inst✝² : CategoryTheory.Category.{?u.22112, u_1} C
          inst✝¹ : CategoryTheory.Preadditive C
          inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
          X : CategoryTheory.SimplicialObject C
          n : Nat
          ⊢ Eq ((AlgebraicTopology.DoldKan.Γ₀.splitting (AlgebraicTopology.AlternatingFa …
        -/
        dsimp [N₁]
        simp only [← Splitting.cofan_inj_id, Splitting.ι_desc, comp_id, Splitting.ι_desc_assoc,
          assoc, PInfty_f_idem_assoc] }
  naturality {X Y} f := by
    /-
      C : Type u_1
      inst✝² : CategoryTheory.Category.{?u.22112, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
      X Y : CategoryTheory.SimplicialObject C
      f : Quiver.Hom X Y
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((AlgebraicTopology.DoldKan.N₁.comp A …
    -/
    ext1
    /-
      case h
      C : Type u_1
      inst✝² : CategoryTheory.Category.{?u.22112, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
      X Y : CategoryTheory.SimplicialObject C
      f : Quiver.Hom X Y
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((AlgebraicTopology.DoldKan.N₁.comp A …
    -/
    apply (Γ₀.splitting K[X]).hom_ext
    /-
      case h.h
      C : Type u_1
      inst✝² : CategoryTheory.Category.{?u.22112, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
      X Y : CategoryTheory.SimplicialObject C
      f : Quiver.Hom X Y
      ⊢ ∀ (n : Nat), Eq ((AlgebraicTopology.DoldKan.Γ₀.splitting (AlgebraicTopology. …
    -/
    intro n
    /-
      case h.h
      C : Type u_1
      inst✝² : CategoryTheory.Category.{?u.22112, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
      X Y : CategoryTheory.SimplicialObject C
      f : Quiver.Hom X Y
      n : Nat
      ⊢ Eq ((AlgebraicTopology.DoldKan.Γ₀.splitting (AlgebraicTopology.AlternatingFa …
    -/
    dsimp [N₁, toKaroubi]
    simp only [← Splitting.cofan_inj_id, Splitting.ι_desc, Splitting.ι_desc_assoc, assoc,
      PInfty_f_idem_assoc, Karoubi.comp_f, NatTrans.comp_app, Γ₂_map_f_app,
      HomologicalComplex.comp_f, AlternatingFaceMapComplex.map_f, PInfty_f_naturality_assoc,
      NatTrans.naturality, Splitting.IndexSet.id_fst, unop_op, len_mk]


/-- The compatibility isomorphism relating `N₂ ⋙ Γ₂` and `N₁ ⋙ Γ₂`. -/
def Γ₂N₂ToKaroubiIso : toKaroubi (SimplicialObject C) ⋙ N₂ ⋙ Γ₂ ≅ N₁ ⋙ Γ₂ :=
  (Functor.associator _ _ _).symm ≪≫ isoWhiskerRight toKaroubiCompN₂IsoN₁ Γ₂


@[simp]
lemma Γ₂N₂ToKaroubiIso_hom_app (X : SimplicialObject C) :
    Γ₂N₂ToKaroubiIso.hom.app X = Γ₂.map (toKaroubiCompN₂IsoN₁.hom.app X) := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
    X : CategoryTheory.SimplicialObject C
    ⊢ Eq (AlgebraicTopology.DoldKan.Γ₂N₂ToKaroubiIso.hom.app X) (AlgebraicTopology …
  -/
  simp [Γ₂N₂ToKaroubiIso]
  /-
    🎉 no goals
  -/


@[simp]
lemma Γ₂N₂ToKaroubiIso_inv_app (X : SimplicialObject C) :
    Γ₂N₂ToKaroubiIso.inv.app X = Γ₂.map (toKaroubiCompN₂IsoN₁.inv.app X) := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
    X : CategoryTheory.SimplicialObject C
    ⊢ Eq (AlgebraicTopology.DoldKan.Γ₂N₂ToKaroubiIso.inv.app X) (AlgebraicTopology …
  -/
  simp [Γ₂N₂ToKaroubiIso]
  /-
    🎉 no goals
  -/


/-- The natural transformation `N₂ ⋙ Γ₂ ⟶ 𝟭 (SimplicialObject C)`. -/
def natTrans : (N₂ : Karoubi (SimplicialObject C) ⥤ _) ⋙ Γ₂ ⟶ 𝟭 _ :=
  ((whiskeringLeft _ _ _).obj (toKaroubi (SimplicialObject C))).preimage
    (Γ₂N₂ToKaroubiIso.hom ≫ Γ₂N₁.natTrans)


theorem natTrans_app_f_app (P : Karoubi (SimplicialObject C)) :
    Γ₂N₂.natTrans.app P =
      (N₂ ⋙ Γ₂).map P.decompId_i ≫
        (Γ₂N₂ToKaroubiIso.hom ≫ Γ₂N₁.natTrans).app P.X ≫ P.decompId_p := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
    P : CategoryTheory.Idempotents.Karoubi (CategoryTheory.SimplicialObject C)
    ⊢ Eq (AlgebraicTopology.DoldKan.Γ₂N₂.natTrans.app P) (CategoryTheory.CategoryS …
  -/
  dsimp only [natTrans]
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
    P : CategoryTheory.Idempotents.Karoubi (CategoryTheory.SimplicialObject C)
    ⊢ Eq ((((CategoryTheory.whiskeringLeft (CategoryTheory.SimplicialObject C) (Ca …
  -/
  simp only [whiskeringLeft_obj_preimage_app, Functor.id_map, assoc]
  /-
    🎉 no goals
  -/


theorem compatibility_Γ₂N₁_Γ₂N₂_natTrans (X : SimplicialObject C) :
    Γ₂N₁.natTrans.app X =
      (Γ₂N₂ToKaroubiIso.app X).inv ≫
        Γ₂N₂.natTrans.app ((toKaroubi (SimplicialObject C)).obj X) := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
    X : CategoryTheory.SimplicialObject C
    ⊢ Eq (AlgebraicTopology.DoldKan.Γ₂N₁.natTrans.app X) (CategoryTheory.CategoryS …
  -/
  rw [Γ₂N₂.natTrans_app_f_app]
  dsimp only [Karoubi.decompId_i_toKaroubi, Karoubi.decompId_p_toKaroubi, Functor.comp_map,
    NatTrans.comp_app]
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
    X : CategoryTheory.SimplicialObject C
    ⊢ Eq (AlgebraicTopology.DoldKan.Γ₂N₁.natTrans.app X) (CategoryTheory.CategoryS …
  -/
  rw [N₂.map_id, Γ₂.map_id, Iso.app_inv]
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
    X : CategoryTheory.SimplicialObject C
    ⊢ Eq (AlgebraicTopology.DoldKan.Γ₂N₁.natTrans.app X) (CategoryTheory.CategoryS …
  -/
  dsimp only [toKaroubi]
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
    X : CategoryTheory.SimplicialObject C
    ⊢ Eq (AlgebraicTopology.DoldKan.Γ₂N₁.natTrans.app X) (CategoryTheory.CategoryS …
  -/
  erw [id_comp]
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
    X : CategoryTheory.SimplicialObject C
    ⊢ Eq (AlgebraicTopology.DoldKan.Γ₂N₁.natTrans.app X) (CategoryTheory.CategoryS …
  -/
  rw [comp_id, Iso.inv_hom_id_app_assoc]
  /-
    🎉 no goals
  -/


theorem identity_N₂_objectwise (P : Karoubi (SimplicialObject C)) :
    (N₂Γ₂.inv.app (N₂.obj P) : N₂.obj P ⟶ N₂.obj (Γ₂.obj (N₂.obj P))) ≫
    N₂.map (Γ₂N₂.natTrans.app P) = 𝟙 (N₂.obj P) := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
    P : CategoryTheory.Idempotents.Karoubi (CategoryTheory.SimplicialObject C)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicTopology.DoldKan.N₂Γ₂.inv.a …
  -/
  ext n
  have eq₁ : (N₂Γ₂.inv.app (N₂.obj P)).f.f n = PInfty.f n ≫ P.p.app (op [n]) ≫
      ((Γ₀.splitting (N₂.obj P).X).cofan _).inj (Splitting.IndexSet.id (op [n])) := by
    simp only [N₂Γ₂_inv_app_f_f, N₂_obj_p_f, assoc]
  have eq₂ : ((Γ₀.splitting (N₂.obj P).X).cofan _).inj (Splitting.IndexSet.id (op [n])) ≫
      (N₂.map (Γ₂N₂.natTrans.app P)).f.f n = PInfty.f n ≫ P.p.app (op [n]) := by
    dsimp
    rw [PInfty_on_Γ₀_splitting_summand_eq_self_assoc, Γ₂N₂.natTrans_app_f_app]
    dsimp
    rw [Γ₂N₂ToKaroubiIso_hom_app, assoc, Splitting.ι_desc_assoc, assoc, assoc]
    dsimp [toKaroubi]
    rw [Splitting.ι_desc_assoc]
    dsimp
    simp only [assoc, Splitting.ι_desc_assoc, unop_op, Splitting.IndexSet.id_fst,
      len_mk, NatTrans.naturality, PInfty_f_idem_assoc,
      PInfty_f_naturality_assoc, app_idem_assoc]
    erw [P.X.map_id, comp_id]
  simp only [Karoubi.comp_f, HomologicalComplex.comp_f, Karoubi.id_f, N₂_obj_p_f, assoc,
    eq₁, eq₂, PInfty_f_naturality_assoc, app_idem, PInfty_f_idem_assoc]

-- Porting note: `Functor.associator` was added to the statement in order to prevent a timeout

theorem identity_N₂ :
    (𝟙 (N₂ : Karoubi (SimplicialObject C) ⥤ _) ◫ N₂Γ₂.inv) ≫
    (Functor.associator _ _ _).inv ≫ Γ₂N₂.natTrans ◫ 𝟙 (@N₂ C _ _) = 𝟙 N₂ := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.NatTrans.hcomp (Categ …
  -/
  ext P : 2
  dsimp only [NatTrans.comp_app, NatTrans.hcomp_app, Functor.comp_map, Functor.associator,
    NatTrans.id_app, Functor.comp_obj]
  /-
    case w.h
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
    P : CategoryTheory.Idempotents.Karoubi (CategoryTheory.SimplicialObject C)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  rw [Γ₂.map_id, N₂.map_id, comp_id, id_comp, id_comp, identity_N₂_objectwise P]
  /-
    🎉 no goals
  -/


instance : IsIso (Γ₂N₂.natTrans : (N₂ : Karoubi (SimplicialObject C) ⥤ _) ⋙ _ ⟶ _) := by
  have : ∀ P : Karoubi (SimplicialObject C), IsIso (Γ₂N₂.natTrans.app P) := by
    intro P
    have : IsIso (N₂.map (Γ₂N₂.natTrans.app P)) := by
      have h := identity_N₂_objectwise P
      erw [hom_comp_eq_id] at h
      rw [h]
      infer_instance
    exact isIso_of_reflects_iso _ N₂
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
    this : ∀ (P : CategoryTheory.Idempotents.Karoubi (CategoryTheory.SimplicialObj …
    ⊢ CategoryTheory.IsIso AlgebraicTopology.DoldKan.Γ₂N₂.natTrans
  -/
  apply NatIso.isIso_of_isIso_app
  /-
    🎉 no goals
  -/


instance : IsIso (Γ₂N₁.natTrans : (N₁ : SimplicialObject C ⥤ _) ⋙ _ ⟶ _) := by
  have : ∀ X : SimplicialObject C, IsIso (Γ₂N₁.natTrans.app X) := by
    intro X
    rw [compatibility_Γ₂N₁_Γ₂N₂_natTrans]
    infer_instance
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
    this : ∀ (X : CategoryTheory.SimplicialObject C), CategoryTheory.IsIso (Algebr …
    ⊢ CategoryTheory.IsIso AlgebraicTopology.DoldKan.Γ₂N₁.natTrans
  -/
  apply NatIso.isIso_of_isIso_app
  /-
    🎉 no goals
  -/


/-- The unit isomorphism of the Dold-Kan equivalence. -/
@[simps! inv]
def Γ₂N₂ : 𝟭 _ ≅ (N₂ : Karoubi (SimplicialObject C) ⥤ _) ⋙ Γ₂ :=
  (asIso Γ₂N₂.natTrans).symm


/-- The natural isomorphism `toKaroubi (SimplicialObject C) ≅ N₁ ⋙ Γ₂`. -/
@[simps! inv]
def Γ₂N₁ : toKaroubi _ ≅ (N₁ : SimplicialObject C ⥤ _) ⋙ Γ₂ :=
  (asIso Γ₂N₁.natTrans).symm


