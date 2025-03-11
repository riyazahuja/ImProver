/-- The `j`th-iteration of a function `φ : I → I` when `j : J` belongs to
a well-ordered type. -/
noncomputable def transfiniteIterate (j : J) : I → I :=
  SuccOrder.limitRecOn j
    (fun _ _ ↦ id) (fun _ _ g ↦ φ ∘ g) (fun y _ h ↦ ⨆ (x : Set.Iio y), h x.1 x.2)


@[simp]
lemma transfiniteIterate_bot [OrderBot J] (i₀ : I) :
    transfiniteIterate φ (⊥ : J) i₀ = i₀ := by
  /-
    I : Type u
    inst✝⁴ : SupSet I
    φ : I → I
    J : Type w
    inst✝³ : LinearOrder J
    inst✝² : SuccOrder J
    inst✝¹ : WellFoundedLT J
    inst✝ : OrderBot J
    i₀ : I
    ⊢ Eq (transfiniteIterate φ Bot.bot i₀) i₀
  -/
  dsimp [transfiniteIterate]
  /-
    I : Type u
    inst✝⁴ : SupSet I
    φ : I → I
    J : Type w
    inst✝³ : LinearOrder J
    inst✝² : SuccOrder J
    inst✝¹ : WellFoundedLT J
    inst✝ : OrderBot J
    i₀ : I
    ⊢ Eq (SuccOrder.limitRecOn Bot.bot (fun x x => id) (fun x x g => Function.comp …
  -/
  simp only [isMin_iff_eq_bot, SuccOrder.limitRecOn_isMin, id_eq]
  /-
    🎉 no goals
  -/


lemma transfiniteIterate_succ (i₀ : I) (j : J) (hj : ¬ IsMax j):
    transfiniteIterate φ (Order.succ j) i₀ =
      φ (transfiniteIterate φ j i₀) := by
  /-
    I : Type u
    inst✝³ : SupSet I
    φ : I → I
    J : Type w
    inst✝² : LinearOrder J
    inst✝¹ : SuccOrder J
    inst✝ : WellFoundedLT J
    i₀ : I
    j : J
    hj : Not (IsMax j)
    ⊢ Eq (transfiniteIterate φ (Order.succ j) i₀) (φ (transfiniteIterate φ j i₀))
  -/
  dsimp [transfiniteIterate]
  /-
    I : Type u
    inst✝³ : SupSet I
    φ : I → I
    J : Type w
    inst✝² : LinearOrder J
    inst✝¹ : SuccOrder J
    inst✝ : WellFoundedLT J
    i₀ : I
    j : J
    hj : Not (IsMax j)
    ⊢ Eq (SuccOrder.limitRecOn (Order.succ j) (fun x x => id) (fun x x g => Functi …
  -/
  rw [SuccOrder.limitRecOn_succ_of_not_isMax _ _ _ hj]
  /-
    I : Type u
    inst✝³ : SupSet I
    φ : I → I
    J : Type w
    inst✝² : LinearOrder J
    inst✝¹ : SuccOrder J
    inst✝ : WellFoundedLT J
    i₀ : I
    j : J
    hj : Not (IsMax j)
    ⊢ Eq (Function.comp φ (SuccOrder.limitRecOn j (fun x x => id) (fun x x g => Fu …
  -/
  rfl
  /-
    🎉 no goals
  -/


lemma transfiniteIterate_limit (i₀ : I) (j : J) (hj : Order.IsSuccLimit j) :
    transfiniteIterate φ j i₀ =
      ⨆ (x : Set.Iio j), transfiniteIterate φ x.1 i₀ := by
  /-
    I : Type u
    inst✝³ : SupSet I
    φ : I → I
    J : Type w
    inst✝² : LinearOrder J
    inst✝¹ : SuccOrder J
    inst✝ : WellFoundedLT J
    i₀ : I
    j : J
    hj : Order.IsSuccLimit j
    ⊢ Eq (transfiniteIterate φ j i₀) (iSup fun x => transfiniteIterate φ (↑x) i₀)
  -/
  dsimp [transfiniteIterate]
  /-
    I : Type u
    inst✝³ : SupSet I
    φ : I → I
    J : Type w
    inst✝² : LinearOrder J
    inst✝¹ : SuccOrder J
    inst✝ : WellFoundedLT J
    i₀ : I
    j : J
    hj : Order.IsSuccLimit j
    ⊢ Eq (SuccOrder.limitRecOn j (fun x x => id) (fun x x g => Function.comp φ g)  …
  -/
  rw [SuccOrder.limitRecOn_of_isSuccLimit _ _ _ hj]
  /-
    I : Type u
    inst✝³ : SupSet I
    φ : I → I
    J : Type w
    inst✝² : LinearOrder J
    inst✝¹ : SuccOrder J
    inst✝ : WellFoundedLT J
    i₀ : I
    j : J
    hj : Order.IsSuccLimit j
    ⊢ Eq (iSup (fun x => (fun x x_1 => SuccOrder.limitRecOn x (fun x x => id) (fun …
  -/
  simp only [iSup_apply]
  /-
    🎉 no goals
  -/


lemma monotone_transfiniteIterate (hφ : ∀ (i : I), i ≤ φ i) :
    Monotone (fun (j : J) ↦ transfiniteIterate φ j i₀) := by
  /-
    I : Type u
    inst✝⁴ : CompleteLattice I
    φ : I → I
    i₀ : I
    J : Type w
    inst✝³ : LinearOrder J
    inst✝² : OrderBot J
    inst✝¹ : SuccOrder J
    inst✝ : WellFoundedLT J
    hφ : ∀ (i : I), LE.le i (φ i)
    ⊢ Monotone fun j => transfiniteIterate φ j i₀
  -/
  intro k j hkj
  induction j using SuccOrder.limitRecOn with
  | hm k hk =>
      obtain rfl := hk.eq_bot
      obtain rfl : k = ⊥ := by simpa using hkj
      rfl
  | hs k' hk' hkk' =>
      obtain hkj | rfl := hkj.lt_or_eq
      · refine (hkk' ((Order.lt_succ_iff_of_not_isMax hk').mp hkj)).trans ?_
        dsimp
        rw [transfiniteIterate_succ _ _ _ hk']
        apply hφ
      · rfl
  | hl k' hk' _ =>
      obtain hkj | rfl := hkj.lt_or_eq
      · dsimp
        rw [transfiniteIterate_limit _ _ _ hk']
        exact le_iSup (fun (⟨l, hl⟩ : Set.Iio k') ↦ transfiniteIterate φ l i₀) ⟨k, hkj⟩
      · rfl


lemma top_mem_range_transfiniteIterate
    (hφ' : ∀ (i : I) (_ : i ≠ ⊤), i < φ i) (φtop : φ ⊤ = ⊤)
    (H : ¬ Function.Injective (fun (j : J) ↦ transfiniteIterate φ j i₀)) :
    ∃ (j : J), transfiniteIterate φ j i₀ = ⊤ := by
  have hφ (i : I) : i ≤ φ i := by
    by_cases hi : i = ⊤
    · subst hi
      rw [φtop]
    · exact (hφ' i hi).le
  obtain ⟨j₁, j₂, hj, eq⟩ : ∃ (j₁ j₂ : J) (hj : j₁ < j₂),
      transfiniteIterate φ j₁ i₀ = transfiniteIterate φ j₂ i₀ := by
    dsimp [Function.Injective] at H
    simp only [not_forall] at H
    obtain ⟨j₁, j₂, eq, hj⟩ := H
    by_cases hj' : j₁ < j₂
    · exact ⟨j₁, j₂, hj', eq⟩
    · simp only [not_lt] at hj'
      obtain hj' | rfl := hj'.lt_or_eq
      · exact ⟨j₂, j₁, hj', eq.symm⟩
      · simp at hj
  /-
    case intro.intro.intro
    I : Type u
    inst✝⁴ : CompleteLattice I
    φ : I → I
    i₀ : I
    J : Type w
    inst✝³ : LinearOrder J
    inst✝² : OrderBot J
    inst✝¹ : SuccOrder J
    inst✝ : WellFoundedLT J
    hφ' : ∀ (i : I), Ne i Top.top → LT.lt i (φ i)
    φtop : Eq (φ Top.top) Top.top
    H : Not (Function.Injective fun j => transfiniteIterate φ j i₀)
    hφ : ∀ (i : I), LE.le i (φ i)
    j₁ j₂ : J
    hj : LT.lt j₁ j₂
    eq : Eq (transfiniteIterate φ j₁ i₀) (transfiniteIterate φ j₂ i₀)
    ⊢ Exists fun j => Eq (transfiniteIterate φ j i₀) Top.top
  -/
  by_contra!
  suffices transfiniteIterate φ j₁ i₀ < transfiniteIterate φ j₂ i₀ by
    simp only [eq, lt_self_iff_false] at this
  have hj₁ : ¬ IsMax j₁ := by
    simp only [not_isMax_iff]
    exact ⟨_, hj⟩
  /-
    case intro.intro.intro
    I : Type u
    inst✝⁴ : CompleteLattice I
    φ : I → I
    i₀ : I
    J : Type w
    inst✝³ : LinearOrder J
    inst✝² : OrderBot J
    inst✝¹ : SuccOrder J
    inst✝ : WellFoundedLT J
    hφ' : ∀ (i : I), Ne i Top.top → LT.lt i (φ i)
    φtop : Eq (φ Top.top) Top.top
    H : Not (Function.Injective fun j => transfiniteIterate φ j i₀)
    hφ : ∀ (i : I), LE.le i (φ i)
    j₁ j₂ : J
    hj : LT.lt j₁ j₂
    eq : Eq (transfiniteIterate φ j₁ i₀) (transfiniteIterate φ j₂ i₀)
    this : ∀ (j : J), Ne (transfiniteIterate φ j i₀) Top.top
    hj₁ : Not (IsMax j₁)
    ⊢ LT.lt (transfiniteIterate φ j₁ i₀) (transfiniteIterate φ j₂ i₀)
  -/
  refine lt_of_lt_of_le (hφ' _ (this j₁)) ?_
  /-
    case intro.intro.intro
    I : Type u
    inst✝⁴ : CompleteLattice I
    φ : I → I
    i₀ : I
    J : Type w
    inst✝³ : LinearOrder J
    inst✝² : OrderBot J
    inst✝¹ : SuccOrder J
    inst✝ : WellFoundedLT J
    hφ' : ∀ (i : I), Ne i Top.top → LT.lt i (φ i)
    φtop : Eq (φ Top.top) Top.top
    H : Not (Function.Injective fun j => transfiniteIterate φ j i₀)
    hφ : ∀ (i : I), LE.le i (φ i)
    j₁ j₂ : J
    hj : LT.lt j₁ j₂
    eq : Eq (transfiniteIterate φ j₁ i₀) (transfiniteIterate φ j₂ i₀)
    this : ∀ (j : J), Ne (transfiniteIterate φ j i₀) Top.top
    hj₁ : Not (IsMax j₁)
    ⊢ LE.le (φ (transfiniteIterate φ j₁ i₀)) (transfiniteIterate φ j₂ i₀)
  -/
  rw [← transfiniteIterate_succ φ i₀ j₁ hj₁]
  /-
    case intro.intro.intro
    I : Type u
    inst✝⁴ : CompleteLattice I
    φ : I → I
    i₀ : I
    J : Type w
    inst✝³ : LinearOrder J
    inst✝² : OrderBot J
    inst✝¹ : SuccOrder J
    inst✝ : WellFoundedLT J
    hφ' : ∀ (i : I), Ne i Top.top → LT.lt i (φ i)
    φtop : Eq (φ Top.top) Top.top
    H : Not (Function.Injective fun j => transfiniteIterate φ j i₀)
    hφ : ∀ (i : I), LE.le i (φ i)
    j₁ j₂ : J
    hj : LT.lt j₁ j₂
    eq : Eq (transfiniteIterate φ j₁ i₀) (transfiniteIterate φ j₂ i₀)
    this : ∀ (j : J), Ne (transfiniteIterate φ j i₀) Top.top
    hj₁ : Not (IsMax j₁)
    ⊢ LE.le (transfiniteIterate φ (Order.succ j₁) i₀) (transfiniteIterate φ j₂ i₀)
  -/
  exact monotone_transfiniteIterate _ _ hφ (Order.succ_le_of_lt hj)
  /-
    🎉 no goals
  -/


