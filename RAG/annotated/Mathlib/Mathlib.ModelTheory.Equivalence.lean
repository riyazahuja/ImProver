/-- `φ ⟹[T] ψ` indicates that `φ` implies `ψ` in models of `T`. -/
protected def Imp (T : L.Theory) (φ ψ : L.BoundedFormula α n) : Prop :=
  T ⊨ᵇ φ.imp ψ


@[inherit_doc FirstOrder.Language.Theory.Imp]
scoped[FirstOrder] notation:51 φ:50 " ⟹[" T "] " ψ:51 => Language.Theory.Imp T φ ψ


@[refl]
protected theorem refl (φ : L.BoundedFormula α n) : φ ⟹[T] φ := fun _ _ _ => id


instance : IsRefl (L.BoundedFormula α n) T.Imp := ⟨Imp.refl⟩


@[trans]
protected theorem trans {φ ψ θ : L.BoundedFormula α n} (h1 : φ ⟹[T] ψ) (h2 : ψ ⟹[T] θ) :
    φ ⟹[T] θ := fun M v xs => (h2 M v xs) ∘ (h1 M v xs)


instance : IsTrans (L.BoundedFormula α n) T.Imp := ⟨fun _ _ _ => Imp.trans⟩


lemma bot_imp (φ : L.BoundedFormula α n) : ⊥ ⟹[T] φ := fun M v xs => by
  /-
    L : FirstOrder.Language
    T : L.Theory
    α : Type w
    n : Nat
    φ : L.BoundedFormula α n
    M : T.ModelType
    v : α → ↑M
    xs : Fin n → ↑M
    ⊢ (Bot.bot.imp φ).Realize v xs
  -/
  simp only [BoundedFormula.realize_imp, BoundedFormula.realize_bot, false_implies]
  /-
    🎉 no goals
  -/


lemma imp_top (φ : L.BoundedFormula α n) : φ ⟹[T] ⊤ := fun M v xs => by
  /-
    L : FirstOrder.Language
    T : L.Theory
    α : Type w
    n : Nat
    φ : L.BoundedFormula α n
    M : T.ModelType
    v : α → ↑M
    xs : Fin n → ↑M
    ⊢ (φ.imp Top.top).Realize v xs
  -/
  simp only [BoundedFormula.realize_imp, BoundedFormula.realize_top, implies_true]
  /-
    🎉 no goals
  -/


lemma imp_sup_left (φ ψ : L.BoundedFormula α n) : φ ⟹[T] φ ⊔ ψ := fun M v xs => by
  /-
    L : FirstOrder.Language
    T : L.Theory
    α : Type w
    n : Nat
    φ ψ : L.BoundedFormula α n
    M : T.ModelType
    v : α → ↑M
    xs : Fin n → ↑M
    ⊢ (φ.imp (Max.max φ ψ)).Realize v xs
  -/
  simp only [BoundedFormula.realize_imp, BoundedFormula.realize_sup]
  /-
    L : FirstOrder.Language
    T : L.Theory
    α : Type w
    n : Nat
    φ ψ : L.BoundedFormula α n
    M : T.ModelType
    v : α → ↑M
    xs : Fin n → ↑M
    ⊢ φ.Realize v xs → Or (φ.Realize v xs) (ψ.Realize v xs)
  -/
  exact Or.inl
  /-
    🎉 no goals
  -/


lemma imp_sup_right (φ ψ : L.BoundedFormula α n) : ψ ⟹[T] φ ⊔ ψ := fun M v xs => by
  /-
    L : FirstOrder.Language
    T : L.Theory
    α : Type w
    n : Nat
    φ ψ : L.BoundedFormula α n
    M : T.ModelType
    v : α → ↑M
    xs : Fin n → ↑M
    ⊢ (ψ.imp (Max.max φ ψ)).Realize v xs
  -/
  simp only [BoundedFormula.realize_imp, BoundedFormula.realize_sup]
  /-
    L : FirstOrder.Language
    T : L.Theory
    α : Type w
    n : Nat
    φ ψ : L.BoundedFormula α n
    M : T.ModelType
    v : α → ↑M
    xs : Fin n → ↑M
    ⊢ ψ.Realize v xs → Or (φ.Realize v xs) (ψ.Realize v xs)
  -/
  exact Or.inr
  /-
    🎉 no goals
  -/


lemma sup_imp {φ ψ θ : L.BoundedFormula α n} (h₁ : φ ⟹[T] θ) (h₂ : ψ ⟹[T] θ) :
    φ ⊔ ψ ⟹[T] θ := fun M v xs => by
  /-
    L : FirstOrder.Language
    T : L.Theory
    α : Type w
    n : Nat
    φ ψ θ : L.BoundedFormula α n
    h₁ : T.Imp φ θ
    h₂ : T.Imp ψ θ
    M : T.ModelType
    v : α → ↑M
    xs : Fin n → ↑M
    ⊢ ((Max.max φ ψ).imp θ).Realize v xs
  -/
  simp only [BoundedFormula.realize_imp, BoundedFormula.realize_sup]
  /-
    L : FirstOrder.Language
    T : L.Theory
    α : Type w
    n : Nat
    φ ψ θ : L.BoundedFormula α n
    h₁ : T.Imp φ θ
    h₂ : T.Imp ψ θ
    M : T.ModelType
    v : α → ↑M
    xs : Fin n → ↑M
    ⊢ Or (φ.Realize v xs) (ψ.Realize v xs) → θ.Realize v xs
  -/
  exact fun h => h.elim (h₁ M v xs) (h₂ M v xs)
  /-
    🎉 no goals
  -/


lemma sup_imp_iff {φ ψ θ : L.BoundedFormula α n} :
    (φ ⊔ ψ ⟹[T] θ) ↔ (φ ⟹[T] θ) ∧ (ψ ⟹[T] θ) :=
  ⟨fun h => ⟨(imp_sup_left _ _).trans h, (imp_sup_right _ _).trans h⟩,
    fun ⟨h₁, h₂⟩ => sup_imp h₁ h₂⟩


lemma inf_imp_left (φ ψ : L.BoundedFormula α n) : φ ⊓ ψ ⟹[T] φ := fun M v xs => by
  /-
    L : FirstOrder.Language
    T : L.Theory
    α : Type w
    n : Nat
    φ ψ : L.BoundedFormula α n
    M : T.ModelType
    v : α → ↑M
    xs : Fin n → ↑M
    ⊢ ((Min.min φ ψ).imp φ).Realize v xs
  -/
  simp only [BoundedFormula.realize_imp, BoundedFormula.realize_inf]
  /-
    L : FirstOrder.Language
    T : L.Theory
    α : Type w
    n : Nat
    φ ψ : L.BoundedFormula α n
    M : T.ModelType
    v : α → ↑M
    xs : Fin n → ↑M
    ⊢ And (φ.Realize v xs) (ψ.Realize v xs) → φ.Realize v xs
  -/
  exact And.left
  /-
    🎉 no goals
  -/


lemma inf_imp_right (φ ψ : L.BoundedFormula α n) : φ ⊓ ψ ⟹[T] ψ := fun M v xs => by
  /-
    L : FirstOrder.Language
    T : L.Theory
    α : Type w
    n : Nat
    φ ψ : L.BoundedFormula α n
    M : T.ModelType
    v : α → ↑M
    xs : Fin n → ↑M
    ⊢ ((Min.min φ ψ).imp ψ).Realize v xs
  -/
  simp only [BoundedFormula.realize_imp, BoundedFormula.realize_inf]
  /-
    L : FirstOrder.Language
    T : L.Theory
    α : Type w
    n : Nat
    φ ψ : L.BoundedFormula α n
    M : T.ModelType
    v : α → ↑M
    xs : Fin n → ↑M
    ⊢ And (φ.Realize v xs) (ψ.Realize v xs) → ψ.Realize v xs
  -/
  exact And.right
  /-
    🎉 no goals
  -/


lemma imp_inf {φ ψ θ : L.BoundedFormula α n} (h₁ : φ ⟹[T] ψ) (h₂ : φ ⟹[T] θ) :
    φ ⟹[T] ψ ⊓ θ := fun M v xs => by
  /-
    L : FirstOrder.Language
    T : L.Theory
    α : Type w
    n : Nat
    φ ψ θ : L.BoundedFormula α n
    h₁ : T.Imp φ ψ
    h₂ : T.Imp φ θ
    M : T.ModelType
    v : α → ↑M
    xs : Fin n → ↑M
    ⊢ (φ.imp (Min.min ψ θ)).Realize v xs
  -/
  simp only [BoundedFormula.realize_imp, BoundedFormula.realize_inf]
  /-
    L : FirstOrder.Language
    T : L.Theory
    α : Type w
    n : Nat
    φ ψ θ : L.BoundedFormula α n
    h₁ : T.Imp φ ψ
    h₂ : T.Imp φ θ
    M : T.ModelType
    v : α → ↑M
    xs : Fin n → ↑M
    ⊢ φ.Realize v xs → And (ψ.Realize v xs) (θ.Realize v xs)
  -/
  exact fun h => ⟨h₁ M v xs h, h₂ M v xs h⟩
  /-
    🎉 no goals
  -/


lemma imp_inf_iff {φ ψ θ : L.BoundedFormula α n} :
    (φ ⟹[T] ψ ⊓ θ) ↔ (φ ⟹[T] ψ) ∧ (φ ⟹[T] θ) :=
  ⟨fun h => ⟨h.trans (inf_imp_left _ _), h.trans (inf_imp_right _ _)⟩,
    fun ⟨h₁, h₂⟩ => imp_inf h₁ h₂⟩


/-- Two (bounded) formulas are semantically equivalent over a theory `T` when they have the same
interpretation in every model of `T`. (This is also known as logical equivalence, which also has a
proof-theoretic definition.) -/
protected def Iff (T : L.Theory) (φ ψ : L.BoundedFormula α n) : Prop :=
  T ⊨ᵇ φ.iff ψ


@[inherit_doc FirstOrder.Language.Theory.Iff]
scoped[FirstOrder]
notation:51 φ:50 " ⇔[" T "] " ψ:51 => Language.Theory.Iff T φ ψ


theorem iff_iff_imp_and_imp {φ ψ : L.BoundedFormula α n} :
    (φ ⇔[T] ψ) ↔ (φ ⟹[T] ψ) ∧ (ψ ⟹[T] φ) := by
  simp only [Theory.Imp, ModelsBoundedFormula, BoundedFormula.realize_imp, ← forall_and,
    Theory.Iff, BoundedFormula.realize_iff, iff_iff_implies_and_implies]


theorem imp_antisymm {φ ψ : L.BoundedFormula α n} (h₁ : φ ⟹[T] ψ) (h₂ : ψ ⟹[T] φ) :
    φ ⇔[T] ψ :=
  iff_iff_imp_and_imp.2 ⟨h₁, h₂⟩


protected theorem mp {φ ψ : L.BoundedFormula α n} (h : φ ⇔[T] ψ) :
    φ ⟹[T] ψ := (iff_iff_imp_and_imp.1 h).1


protected theorem mpr {φ ψ : L.BoundedFormula α n} (h : φ ⇔[T] ψ) :
    ψ ⟹[T] φ := (iff_iff_imp_and_imp.1 h).2


@[refl]
protected theorem refl (φ : L.BoundedFormula α n) : φ ⇔[T] φ :=
                   /-
                     L : FirstOrder.Language
                     T : L.Theory
                     α : Type w
                     n : Nat
                     φ : L.BoundedFormula α n
                     M : T.ModelType
                     v : α → ↑M
                     xs : Fin n → ↑M
                     ⊢ (φ.iff φ).Realize v xs
                   -/
  fun M v xs => by rw [BoundedFormula.realize_iff]
                   /-
                     🎉 no goals
                   -/


instance : IsRefl (L.BoundedFormula α n) T.Iff :=
  ⟨Iff.refl⟩


@[symm]
protected theorem symm {φ ψ : L.BoundedFormula α n}
    (h : φ ⇔[T] ψ) : ψ ⇔[T] φ := fun M v xs => by
  /-
    L : FirstOrder.Language
    T : L.Theory
    α : Type w
    n : Nat
    φ ψ : L.BoundedFormula α n
    h : T.Iff φ ψ
    M : T.ModelType
    v : α → ↑M
    xs : Fin n → ↑M
    ⊢ (ψ.iff φ).Realize v xs
  -/
  rw [BoundedFormula.realize_iff, Iff.comm, ← BoundedFormula.realize_iff]
  /-
    L : FirstOrder.Language
    T : L.Theory
    α : Type w
    n : Nat
    φ ψ : L.BoundedFormula α n
    h : T.Iff φ ψ
    M : T.ModelType
    v : α → ↑M
    xs : Fin n → ↑M
    ⊢ (φ.iff ψ).Realize v xs
  -/
  exact h M v xs
  /-
    🎉 no goals
  -/


instance : IsSymm (L.BoundedFormula α n) T.Iff :=
  ⟨fun _ _ => Iff.symm⟩


@[trans]
protected theorem trans {φ ψ θ : L.BoundedFormula α n}
    (h1 : φ ⇔[T] ψ) (h2 : ψ ⇔[T] θ) :
    φ ⇔[T] θ := fun M v xs => by
  /-
    L : FirstOrder.Language
    T : L.Theory
    α : Type w
    n : Nat
    φ ψ θ : L.BoundedFormula α n
    h1 : T.Iff φ ψ
    h2 : T.Iff ψ θ
    M : T.ModelType
    v : α → ↑M
    xs : Fin n → ↑M
    ⊢ (φ.iff θ).Realize v xs
  -/
  have h1' := h1 M v xs
  /-
    L : FirstOrder.Language
    T : L.Theory
    α : Type w
    n : Nat
    φ ψ θ : L.BoundedFormula α n
    h1 : T.Iff φ ψ
    h2 : T.Iff ψ θ
    M : T.ModelType
    v : α → ↑M
    xs : Fin n → ↑M
    h1' : (φ.iff ψ).Realize v xs
    ⊢ (φ.iff θ).Realize v xs
  -/
  have h2' := h2 M v xs
  /-
    L : FirstOrder.Language
    T : L.Theory
    α : Type w
    n : Nat
    φ ψ θ : L.BoundedFormula α n
    h1 : T.Iff φ ψ
    h2 : T.Iff ψ θ
    M : T.ModelType
    v : α → ↑M
    xs : Fin n → ↑M
    h1' : (φ.iff ψ).Realize v xs
    h2' : (ψ.iff θ).Realize v xs
    ⊢ (φ.iff θ).Realize v xs
  -/
  rw [BoundedFormula.realize_iff] at *
  /-
    L : FirstOrder.Language
    T : L.Theory
    α : Type w
    n : Nat
    φ ψ θ : L.BoundedFormula α n
    h1 : T.Iff φ ψ
    h2 : T.Iff ψ θ
    M : T.ModelType
    v : α → ↑M
    xs : Fin n → ↑M
    h1' : Iff (φ.Realize v xs) (ψ.Realize v xs)
    h2' : Iff (ψ.Realize v xs) (θ.Realize v xs)
    ⊢ Iff (φ.Realize v xs) (θ.Realize v xs)
  -/
  exact ⟨h2'.1 ∘ h1'.1, h1'.2 ∘ h2'.2⟩
  /-
    🎉 no goals
  -/


instance : IsTrans (L.BoundedFormula α n) T.Iff :=
  ⟨fun _ _ _ => Iff.trans⟩


theorem realize_bd_iff {φ ψ : L.BoundedFormula α n} (h : φ ⇔[T] ψ)
    {v : α → M} {xs : Fin n → M} : φ.Realize v xs ↔ ψ.Realize v xs :=
  BoundedFormula.realize_iff.1 (h.realize_boundedFormula M)


theorem realize_iff {φ ψ : L.Formula α} {M : Type*} [Nonempty M]
    [L.Structure M] [M ⊨ T] (h : φ ⇔[T] ψ) {v : α → M} :
    φ.Realize v ↔ ψ.Realize v :=
  h.realize_bd_iff


theorem models_sentence_iff {φ ψ : L.Sentence} {M : Type*} [Nonempty M]
    [L.Structure M] [M ⊨ T] (h : φ ⇔[T] ψ) :
    M ⊨ φ ↔ M ⊨ ψ :=
  h.realize_iff


protected theorem all {φ ψ : L.BoundedFormula α (n + 1)}
    (h : φ ⇔[T] ψ) : φ.all ⇔[T] ψ.all := by
  simp_rw [Theory.Iff, ModelsBoundedFormula, BoundedFormula.realize_iff,
    BoundedFormula.realize_all]
  /-
    L : FirstOrder.Language
    T : L.Theory
    α : Type w
    n : Nat
    φ ψ : L.BoundedFormula α (HAdd.hAdd n 1)
    h : T.Iff φ ψ
    ⊢ ∀ (M : T.ModelType) (v : α → ↑M) (xs : Fin n → ↑M), Iff (∀ (a : ↑M), φ.Reali …
  -/
  exact fun M v xs => forall_congr' fun a => h.realize_bd_iff
  /-
    🎉 no goals
  -/


protected theorem ex {φ ψ : L.BoundedFormula α (n + 1)} (h : φ ⇔[T] ψ) :
    φ.ex ⇔[T] ψ.ex := by
  simp_rw [Theory.Iff, ModelsBoundedFormula, BoundedFormula.realize_iff,
    BoundedFormula.realize_ex]
  /-
    L : FirstOrder.Language
    T : L.Theory
    α : Type w
    n : Nat
    φ ψ : L.BoundedFormula α (HAdd.hAdd n 1)
    h : T.Iff φ ψ
    ⊢ ∀ (M : T.ModelType) (v : α → ↑M) (xs : Fin n → ↑M), Iff (Exists fun a => φ.R …
  -/
  exact fun M v xs => exists_congr fun a => h.realize_bd_iff
  /-
    🎉 no goals
  -/


protected theorem not {φ ψ : L.BoundedFormula α n} (h : φ ⇔[T] ψ) :
    φ.not ⇔[T] ψ.not := by
  simp_rw [Theory.Iff, ModelsBoundedFormula, BoundedFormula.realize_iff,
    BoundedFormula.realize_not]
  /-
    L : FirstOrder.Language
    T : L.Theory
    α : Type w
    n : Nat
    φ ψ : L.BoundedFormula α n
    h : T.Iff φ ψ
    ⊢ ∀ (M : T.ModelType) (v : α → ↑M) (xs : Fin n → ↑M), Iff (Not (φ.Realize v xs …
  -/
  exact fun M v xs => not_congr h.realize_bd_iff
  /-
    🎉 no goals
  -/


protected theorem imp {φ ψ φ' ψ' : L.BoundedFormula α n} (h : φ ⇔[T] ψ) (h' : φ' ⇔[T] ψ') :
    (φ.imp φ') ⇔[T] (ψ.imp ψ') := by
  simp_rw [Theory.Iff, ModelsBoundedFormula, BoundedFormula.realize_iff,
    BoundedFormula.realize_imp]
  /-
    L : FirstOrder.Language
    T : L.Theory
    α : Type w
    n : Nat
    φ ψ φ' ψ' : L.BoundedFormula α n
    h : T.Iff φ ψ
    h' : T.Iff φ' ψ'
    ⊢ ∀ (M : T.ModelType) (v : α → ↑M) (xs : Fin n → ↑M), Iff (φ.Realize v xs → φ' …
  -/
  exact fun M v xs => imp_congr h.realize_bd_iff h'.realize_bd_iff
  /-
    🎉 no goals
  -/


/-- Semantic equivalence forms an equivalence relation on formulas. -/
def iffSetoid (T : L.Theory) : Setoid (L.BoundedFormula α n) where
  r := T.Iff
  iseqv := ⟨fun _ => refl _, fun {_ _} h => h.symm, fun {_ _ _} h1 h2 => h1.trans h2⟩


theorem iff_not_not : φ ⇔[T] φ.not.not := fun M v xs => by
  /-
    L : FirstOrder.Language
    T : L.Theory
    α : Type w
    n : Nat
    φ : L.BoundedFormula α n
    M : T.ModelType
    v : α → ↑M
    xs : Fin n → ↑M
    ⊢ (φ.iff φ.not.not).Realize v xs
  -/
  simp
  /-
    🎉 no goals
  -/


theorem imp_iff_not_sup : (φ.imp ψ) ⇔[T] (φ.not ⊔ ψ) :=
                   /-
                     L : FirstOrder.Language
                     T : L.Theory
                     α : Type w
                     n : Nat
                     φ ψ : L.BoundedFormula α n
                     M : T.ModelType
                     v : α → ↑M
                     xs : Fin n → ↑M
                     ⊢ ((φ.imp ψ).iff (Max.max φ.not ψ)).Realize v xs
                   -/
  fun M v xs => by simp [imp_iff_not_or]
                   /-
                     🎉 no goals
                   -/


theorem sup_iff_not_inf_not : (φ ⊔ ψ) ⇔[T] (φ.not ⊓ ψ.not).not :=
                   /-
                     L : FirstOrder.Language
                     T : L.Theory
                     α : Type w
                     n : Nat
                     φ ψ : L.BoundedFormula α n
                     M : T.ModelType
                     v : α → ↑M
                     xs : Fin n → ↑M
                     ⊢ ((Max.max φ ψ).iff (Min.min φ.not ψ.not).not).Realize v xs
                   -/
  fun M v xs => by simp [imp_iff_not_or]
                   /-
                     🎉 no goals
                   -/


theorem inf_iff_not_sup_not : (φ ⊓ ψ) ⇔[T] (φ.not ⊔ ψ.not).not :=
                   /-
                     L : FirstOrder.Language
                     T : L.Theory
                     α : Type w
                     n : Nat
                     φ ψ : L.BoundedFormula α n
                     M : T.ModelType
                     v : α → ↑M
                     xs : Fin n → ↑M
                     ⊢ ((Min.min φ ψ).iff (Max.max φ.not ψ.not).not).Realize v xs
                   -/
  fun M v xs => by simp
                   /-
                     🎉 no goals
                   -/


theorem all_iff_not_ex_not (φ : L.BoundedFormula α (n + 1)) :
                                                /-
                                                  L : FirstOrder.Language
                                                  T : L.Theory
                                                  α : Type w
                                                  n : Nat
                                                  φ : L.BoundedFormula α (HAdd.hAdd n 1)
                                                  M : T.ModelType
                                                  v : α → ↑M
                                                  xs : Fin n → ↑M
                                                  ⊢ (φ.all.iff φ.not.ex.not).Realize v xs
                                                -/
    φ.all ⇔[T] φ.not.ex.not := fun M v xs => by simp
                                                /-
                                                  🎉 no goals
                                                -/


theorem ex_iff_not_all_not (φ : L.BoundedFormula α (n + 1)) :
                                                /-
                                                  L : FirstOrder.Language
                                                  T : L.Theory
                                                  α : Type w
                                                  n : Nat
                                                  φ : L.BoundedFormula α (HAdd.hAdd n 1)
                                                  M : T.ModelType
                                                  v : α → ↑M
                                                  xs : Fin n → ↑M
                                                  ⊢ (φ.ex.iff φ.not.all.not).Realize v xs
                                                -/
    φ.ex ⇔[T] φ.not.all.not := fun M v xs => by simp
                                                /-
                                                  🎉 no goals
                                                -/


theorem iff_all_liftAt : φ ⇔[T] (φ.liftAt 1 n).all :=
  fun M v xs => by
  /-
    L : FirstOrder.Language
    T : L.Theory
    α : Type w
    n : Nat
    φ : L.BoundedFormula α n
    M : T.ModelType
    v : α → ↑M
    xs : Fin n → ↑M
    ⊢ (φ.iff (FirstOrder.Language.BoundedFormula.liftAt 1 n φ).all).Realize v xs
  -/
  rw [realize_iff, realize_all_liftAt_one_self]
  /-
    🎉 no goals
  -/


lemma inf_not_iff_bot :
    φ ⊓ ∼φ ⇔[T] ⊥ := fun M v xs => by
  /-
    L : FirstOrder.Language
    T : L.Theory
    α : Type w
    n : Nat
    φ : L.BoundedFormula α n
    M : T.ModelType
    v : α → ↑M
    xs : Fin n → ↑M
    ⊢ ((Min.min φ φ.not).iff Bot.bot).Realize v xs
  -/
  simp only [realize_iff, realize_inf, realize_not, and_not_self, realize_bot]
  /-
    🎉 no goals
  -/


lemma sup_not_iff_top :
    φ ⊔ ∼φ ⇔[T] ⊤ := fun M v xs => by
  /-
    L : FirstOrder.Language
    T : L.Theory
    α : Type w
    n : Nat
    φ : L.BoundedFormula α n
    M : T.ModelType
    v : α → ↑M
    xs : Fin n → ↑M
    ⊢ ((Max.max φ φ.not).iff Top.top).Realize v xs
  -/
  simp only [realize_iff, realize_sup, realize_not, realize_top, iff_true, or_not]
  /-
    🎉 no goals
  -/


theorem iff_not_not : φ ⇔[T] φ.not.not :=
  BoundedFormula.iff_not_not φ


theorem imp_iff_not_sup : (φ.imp ψ) ⇔[T] (φ.not ⊔ ψ) :=
  BoundedFormula.imp_iff_not_sup φ ψ


theorem sup_iff_not_inf_not : (φ ⊔ ψ) ⇔[T] (φ.not ⊓ ψ.not).not :=
  BoundedFormula.sup_iff_not_inf_not φ ψ


theorem inf_iff_not_sup_not : (φ ⊓ ψ) ⇔[T] (φ.not ⊔ ψ.not).not :=
  BoundedFormula.inf_iff_not_sup_not φ ψ


