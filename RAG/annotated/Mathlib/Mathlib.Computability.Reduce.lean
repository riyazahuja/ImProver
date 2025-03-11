/--
`p` is many-one reducible to `q` if there is a computable function translating questions about `p`
to questions about `q`.
-/
def ManyOneReducible {α β} [Primcodable α] [Primcodable β] (p : α → Prop) (q : β → Prop) :=
  ∃ f, Computable f ∧ ∀ a, p a ↔ q (f a)


@[inherit_doc ManyOneReducible]
infixl:1000 " ≤₀ " => ManyOneReducible


theorem ManyOneReducible.mk {α β} [Primcodable α] [Primcodable β] {f : α → β} (q : β → Prop)
    (h : Computable f) : (fun a => q (f a)) ≤₀ q :=
  ⟨f, h, fun _ => Iff.rfl⟩


@[refl]
theorem manyOneReducible_refl {α} [Primcodable α] (p : α → Prop) : p ≤₀ p :=
                         /-
                           α : Type u_1
                           inst✝ : Primcodable α
                           p : α → Prop
                           ⊢ ∀ (a : α), Iff (p a) (p (id a))
                         -/
  ⟨id, Computable.id, by simp⟩
                         /-
                           🎉 no goals
                         -/


@[trans]
theorem ManyOneReducible.trans {α β γ} [Primcodable α] [Primcodable β] [Primcodable γ]
    {p : α → Prop} {q : β → Prop} {r : γ → Prop} : p ≤₀ q → q ≤₀ r → p ≤₀ r
  | ⟨f, c₁, h₁⟩, ⟨g, c₂, h₂⟩ =>
    ⟨g ∘ f, c₂.comp c₁,
                            /-
                              α : Type u_1
                              β : Type u_2
                              γ : Type u_3
                              inst✝² : Primcodable α
                              inst✝¹ : Primcodable β
                              inst✝ : Primcodable γ
                              p : α → Prop
                              q : β → Prop
                              r : γ → Prop
                              f : α → β
                              c₁ : Computable f
                              h₁ : ∀ (a : α), Iff (p a) (q (f a))
                              g : β → γ
                              c₂ : Computable g
                              h₂ : ∀ (a : β), Iff (q a) (r (g a))
                              a : α
                              h : p a
                              ⊢ r (Function.comp g f a)
                            -/
                                              /-
                                                🎉 no goals
                                              -/
      fun a => ⟨fun h => by erw [← h₂, ← h₁]; assumption, fun h => by rwa [h₁, h₂]⟩⟩
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


theorem reflexive_manyOneReducible {α} [Primcodable α] : Reflexive (@ManyOneReducible α α _ _) :=
  manyOneReducible_refl


theorem transitive_manyOneReducible {α} [Primcodable α] : Transitive (@ManyOneReducible α α _ _) :=
  fun _ _ _ => ManyOneReducible.trans


/--
`p` is one-one reducible to `q` if there is an injective computable function translating questions
about `p` to questions about `q`.
-/
def OneOneReducible {α β} [Primcodable α] [Primcodable β] (p : α → Prop) (q : β → Prop) :=
  ∃ f, Computable f ∧ Injective f ∧ ∀ a, p a ↔ q (f a)


@[inherit_doc OneOneReducible]
infixl:1000 " ≤₁ " => OneOneReducible


theorem OneOneReducible.mk {α β} [Primcodable α] [Primcodable β] {f : α → β} (q : β → Prop)
    (h : Computable f) (i : Injective f) : (fun a => q (f a)) ≤₁ q :=
  ⟨f, h, i, fun _ => Iff.rfl⟩


@[refl]
theorem oneOneReducible_refl {α} [Primcodable α] (p : α → Prop) : p ≤₁ p :=
                                       /-
                                         α : Type u_1
                                         inst✝ : Primcodable α
                                         p : α → Prop
                                         ⊢ ∀ (a : α), Iff (p a) (p (id a))
                                       -/
  ⟨id, Computable.id, injective_id, by simp⟩
                                       /-
                                         🎉 no goals
                                       -/


@[trans]
theorem OneOneReducible.trans {α β γ} [Primcodable α] [Primcodable β] [Primcodable γ] {p : α → Prop}
    {q : β → Prop} {r : γ → Prop} : p ≤₁ q → q ≤₁ r → p ≤₁ r
  | ⟨f, c₁, i₁, h₁⟩, ⟨g, c₂, i₂, h₂⟩ =>
    ⟨g ∘ f, c₂.comp c₁, i₂.comp i₁, fun a =>
                   /-
                     α : Type u_1
                     β : Type u_2
                     γ : Type u_3
                     inst✝² : Primcodable α
                     inst✝¹ : Primcodable β
                     inst✝ : Primcodable γ
                     p : α → Prop
                     q : β → Prop
                     r : γ → Prop
                     f : α → β
                     c₁ : Computable f
                     i₁ : Function.Injective f
                     h₁ : ∀ (a : α), Iff (p a) (q (f a))
                     g : β → γ
                     c₂ : Computable g
                     i₂ : Function.Injective g
                     h₂ : ∀ (a : β), Iff (q a) (r (g a))
                     a : α
                     h : p a
                     ⊢ r (Function.comp g f a)
                   -/
                                     /-
                                       🎉 no goals
                                     -/
      ⟨fun h => by erw [← h₂, ← h₁]; assumption, fun h => by rwa [h₁, h₂]⟩⟩
                                                             /-
                                                               🎉 no goals
                                                             -/


theorem OneOneReducible.to_many_one {α β} [Primcodable α] [Primcodable β] {p : α → Prop}
    {q : β → Prop} : p ≤₁ q → p ≤₀ q
  | ⟨f, c, _, h⟩ => ⟨f, c, h⟩


theorem OneOneReducible.of_equiv {α β} [Primcodable α] [Primcodable β] {e : α ≃ β} (q : β → Prop)
    (h : Computable e) : (q ∘ e) ≤₁ q :=
  OneOneReducible.mk _ h e.injective


theorem OneOneReducible.of_equiv_symm {α β} [Primcodable α] [Primcodable β] {e : α ≃ β}
    (q : β → Prop) (h : Computable e.symm) : q ≤₁ (q ∘ e) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : Primcodable α
    inst✝ : Primcodable β
    e : Equiv α β
    q : β → Prop
    h : Computable ⇑e.symm
    ⊢ OneOneReducible q (Function.comp q ⇑e)
  -/
  convert OneOneReducible.of_equiv _ h; funext; simp
                                                /-
                                                  🎉 no goals
                                                -/


theorem reflexive_oneOneReducible {α} [Primcodable α] : Reflexive (@OneOneReducible α α _ _) :=
  oneOneReducible_refl


theorem transitive_oneOneReducible {α} [Primcodable α] : Transitive (@OneOneReducible α α _ _) :=
  fun _ _ _ => OneOneReducible.trans


theorem computable_of_manyOneReducible {p : α → Prop} {q : β → Prop} (h₁ : p ≤₀ q)
    (h₂ : ComputablePred q) : ComputablePred p := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : Primcodable α
    inst✝ : Primcodable β
    p : α → Prop
    q : β → Prop
    h₁ : ManyOneReducible p q
    h₂ : ComputablePred q
    ⊢ ComputablePred p
  -/
  rcases h₁ with ⟨f, c, hf⟩
  /-
    case intro.intro
    α : Type u_1
    β : Type u_2
    inst✝¹ : Primcodable α
    inst✝ : Primcodable β
    p : α → Prop
    q : β → Prop
    h₂ : ComputablePred q
    f : α → β
    c : Computable f
    hf : ∀ (a : α), Iff (p a) (q (f a))
    ⊢ ComputablePred p
  -/
  rw [show p = fun a => q (f a) from Set.ext hf]
  /-
    case intro.intro
    α : Type u_1
    β : Type u_2
    inst✝¹ : Primcodable α
    inst✝ : Primcodable β
    p : α → Prop
    q : β → Prop
    h₂ : ComputablePred q
    f : α → β
    c : Computable f
    hf : ∀ (a : α), Iff (p a) (q (f a))
    ⊢ ComputablePred fun a => q (f a)
  -/
  rcases computable_iff.1 h₂ with ⟨g, hg, rfl⟩
  /-
    case intro.intro.intro.intro
    α : Type u_1
    β : Type u_2
    inst✝¹ : Primcodable α
    inst✝ : Primcodable β
    p : α → Prop
    f : α → β
    c : Computable f
    g : β → Bool
    hg : Computable g
    h₂ : ComputablePred fun a => Eq (g a) Bool.true
    hf : ∀ (a : α), Iff (p a) ((fun a => Eq (g a) Bool.true) (f a))
    ⊢ ComputablePred fun a => (fun a => Eq (g a) Bool.true) (f a)
  -/
  exact ⟨by infer_instance, by simpa using hg.comp c⟩
  /-
    🎉 no goals
  -/


theorem computable_of_oneOneReducible {p : α → Prop} {q : β → Prop} (h : p ≤₁ q) :
    ComputablePred q → ComputablePred p :=
  computable_of_manyOneReducible h.to_many_one


/-- `p` and `q` are many-one equivalent if each one is many-one reducible to the other. -/
def ManyOneEquiv {α β} [Primcodable α] [Primcodable β] (p : α → Prop) (q : β → Prop) :=
  p ≤₀ q ∧ q ≤₀ p


/-- `p` and `q` are one-one equivalent if each one is one-one reducible to the other. -/
def OneOneEquiv {α β} [Primcodable α] [Primcodable β] (p : α → Prop) (q : β → Prop) :=
  p ≤₁ q ∧ q ≤₁ p


@[refl]
theorem manyOneEquiv_refl {α} [Primcodable α] (p : α → Prop) : ManyOneEquiv p p :=
  ⟨manyOneReducible_refl _, manyOneReducible_refl _⟩


@[symm]
theorem ManyOneEquiv.symm {α β} [Primcodable α] [Primcodable β] {p : α → Prop} {q : β → Prop} :
    ManyOneEquiv p q → ManyOneEquiv q p :=
  And.symm


@[trans]
theorem ManyOneEquiv.trans {α β γ} [Primcodable α] [Primcodable β] [Primcodable γ] {p : α → Prop}
    {q : β → Prop} {r : γ → Prop} : ManyOneEquiv p q → ManyOneEquiv q r → ManyOneEquiv p r
  | ⟨pq, qp⟩, ⟨qr, rq⟩ => ⟨pq.trans qr, rq.trans qp⟩


theorem equivalence_of_manyOneEquiv {α} [Primcodable α] : Equivalence (@ManyOneEquiv α α _ _) :=
  ⟨manyOneEquiv_refl, fun {_ _} => ManyOneEquiv.symm, fun {_ _ _} => ManyOneEquiv.trans⟩


@[refl]
theorem oneOneEquiv_refl {α} [Primcodable α] (p : α → Prop) : OneOneEquiv p p :=
  ⟨oneOneReducible_refl _, oneOneReducible_refl _⟩


@[symm]
theorem OneOneEquiv.symm {α β} [Primcodable α] [Primcodable β] {p : α → Prop} {q : β → Prop} :
    OneOneEquiv p q → OneOneEquiv q p :=
  And.symm


@[trans]
theorem OneOneEquiv.trans {α β γ} [Primcodable α] [Primcodable β] [Primcodable γ] {p : α → Prop}
    {q : β → Prop} {r : γ → Prop} : OneOneEquiv p q → OneOneEquiv q r → OneOneEquiv p r
  | ⟨pq, qp⟩, ⟨qr, rq⟩ => ⟨pq.trans qr, rq.trans qp⟩


theorem equivalence_of_oneOneEquiv {α} [Primcodable α] : Equivalence (@OneOneEquiv α α _ _) :=
  ⟨oneOneEquiv_refl, fun {_ _} => OneOneEquiv.symm, fun {_ _ _} => OneOneEquiv.trans⟩


theorem OneOneEquiv.to_many_one {α β} [Primcodable α] [Primcodable β] {p : α → Prop}
    {q : β → Prop} : OneOneEquiv p q → ManyOneEquiv p q
  | ⟨pq, qp⟩ => ⟨pq.to_many_one, qp.to_many_one⟩


/-- a computable bijection -/
nonrec def Equiv.Computable {α β} [Primcodable α] [Primcodable β] (e : α ≃ β) :=
  Computable e ∧ Computable e.symm


theorem Equiv.Computable.symm {α β} [Primcodable α] [Primcodable β] {e : α ≃ β} :
    e.Computable → e.symm.Computable :=
  And.symm


theorem Equiv.Computable.trans {α β γ} [Primcodable α] [Primcodable β] [Primcodable γ] {e₁ : α ≃ β}
    {e₂ : β ≃ γ} : e₁.Computable → e₂.Computable → (e₁.trans e₂).Computable
  | ⟨l₁, r₁⟩, ⟨l₂, r₂⟩ => ⟨l₂.comp l₁, r₁.comp r₂⟩


theorem Computable.eqv (α) [Denumerable α] : (Denumerable.eqv α).Computable :=
  ⟨Computable.encode, Computable.ofNat _⟩


theorem Computable.equiv₂ (α β) [Denumerable α] [Denumerable β] :
    (Denumerable.equiv₂ α β).Computable :=
  (Computable.eqv _).trans (Computable.eqv _).symm


theorem OneOneEquiv.of_equiv {α β} [Primcodable α] [Primcodable β] {e : α ≃ β} (h : e.Computable)
    {p} : OneOneEquiv (p ∘ e) p :=
  ⟨OneOneReducible.of_equiv _ h.1, OneOneReducible.of_equiv_symm _ h.2⟩


theorem ManyOneEquiv.of_equiv {α β} [Primcodable α] [Primcodable β] {e : α ≃ β} (h : e.Computable)
    {p} : ManyOneEquiv (p ∘ e) p :=
  (OneOneEquiv.of_equiv h).to_many_one


theorem ManyOneEquiv.le_congr_left {α β γ} [Primcodable α] [Primcodable β] [Primcodable γ]
    {p : α → Prop} {q : β → Prop} {r : γ → Prop} (h : ManyOneEquiv p q) : p ≤₀ r ↔ q ≤₀ r :=
  ⟨h.2.trans, h.1.trans⟩


theorem ManyOneEquiv.le_congr_right {α β γ} [Primcodable α] [Primcodable β] [Primcodable γ]
    {p : α → Prop} {q : β → Prop} {r : γ → Prop} (h : ManyOneEquiv q r) : p ≤₀ q ↔ p ≤₀ r :=
  ⟨fun h' => h'.trans h.1, fun h' => h'.trans h.2⟩


theorem OneOneEquiv.le_congr_left {α β γ} [Primcodable α] [Primcodable β] [Primcodable γ]
    {p : α → Prop} {q : β → Prop} {r : γ → Prop} (h : OneOneEquiv p q) : p ≤₁ r ↔ q ≤₁ r :=
  ⟨h.2.trans, h.1.trans⟩


theorem OneOneEquiv.le_congr_right {α β γ} [Primcodable α] [Primcodable β] [Primcodable γ]
    {p : α → Prop} {q : β → Prop} {r : γ → Prop} (h : OneOneEquiv q r) : p ≤₁ q ↔ p ≤₁ r :=
  ⟨fun h' => h'.trans h.1, fun h' => h'.trans h.2⟩


theorem ManyOneEquiv.congr_left {α β γ} [Primcodable α] [Primcodable β] [Primcodable γ]
    {p : α → Prop} {q : β → Prop} {r : γ → Prop} (h : ManyOneEquiv p q) :
    ManyOneEquiv p r ↔ ManyOneEquiv q r :=
  and_congr h.le_congr_left h.le_congr_right


theorem ManyOneEquiv.congr_right {α β γ} [Primcodable α] [Primcodable β] [Primcodable γ]
    {p : α → Prop} {q : β → Prop} {r : γ → Prop} (h : ManyOneEquiv q r) :
    ManyOneEquiv p q ↔ ManyOneEquiv p r :=
  and_congr h.le_congr_right h.le_congr_left


theorem OneOneEquiv.congr_left {α β γ} [Primcodable α] [Primcodable β] [Primcodable γ]
    {p : α → Prop} {q : β → Prop} {r : γ → Prop} (h : OneOneEquiv p q) :
    OneOneEquiv p r ↔ OneOneEquiv q r :=
  and_congr h.le_congr_left h.le_congr_right


theorem OneOneEquiv.congr_right {α β γ} [Primcodable α] [Primcodable β] [Primcodable γ]
    {p : α → Prop} {q : β → Prop} {r : γ → Prop} (h : OneOneEquiv q r) :
    OneOneEquiv p q ↔ OneOneEquiv p r :=
  and_congr h.le_congr_right h.le_congr_left


@[simp]
theorem ULower.down_computable {α} [Primcodable α] : (ULower.equiv α).Computable :=
  ⟨Primrec.ulower_down.to_comp, Primrec.ulower_up.to_comp⟩


theorem manyOneEquiv_up {α} [Primcodable α] {p : α → Prop} : ManyOneEquiv (p ∘ ULower.up) p :=
  ManyOneEquiv.of_equiv ULower.down_computable.symm


local infixl:1001 " ⊕' " => Sum.elim


theorem OneOneReducible.disjoin_left {α β} [Primcodable α] [Primcodable β] {p : α → Prop}
    {q : β → Prop} : p ≤₁ p ⊕' q :=
  ⟨Sum.inl, Computable.sum_inl, fun _ _ => Sum.inl.inj_iff.1, fun _ => Iff.rfl⟩


theorem OneOneReducible.disjoin_right {α β} [Primcodable α] [Primcodable β] {p : α → Prop}
    {q : β → Prop} : q ≤₁ p ⊕' q :=
  ⟨Sum.inr, Computable.sum_inr, fun _ _ => Sum.inr.inj_iff.1, fun _ => Iff.rfl⟩


theorem disjoin_manyOneReducible {α β γ} [Primcodable α] [Primcodable β] [Primcodable γ]
    {p : α → Prop} {q : β → Prop} {r : γ → Prop} : p ≤₀ r → q ≤₀ r → (p ⊕' q) ≤₀ r
  | ⟨f, c₁, h₁⟩, ⟨g, c₂, h₂⟩ =>
    ⟨Sum.elim f g,
      Computable.id.sum_casesOn (c₁.comp Computable.snd).to₂ (c₂.comp Computable.snd).to₂,
                  /-
                    α : Type u_1
                    β : Type u_2
                    γ : Type u_3
                    inst✝² : Primcodable α
                    inst✝¹ : Primcodable β
                    inst✝ : Primcodable γ
                    p : α → Prop
                    q : β → Prop
                    r : γ → Prop
                    f : α → γ
                    c₁ : Computable f
                    h₁ : ∀ (a : α), Iff (p a) (r (f a))
                    g : β → γ
                    c₂ : Computable g
                    h₂ : ∀ (a : β), Iff (q a) (r (g a))
                    x : Sum α β
                    ⊢ Iff (Sum.elim p q x) (r (Sum.elim f g x))
                  -/
      fun x => by cases x <;> [apply h₁; apply h₂]⟩
                  /-
                    🎉 no goals
                  -/


theorem disjoin_le {α β γ} [Primcodable α] [Primcodable β] [Primcodable γ] {p : α → Prop}
    {q : β → Prop} {r : γ → Prop} : (p ⊕' q) ≤₀ r ↔ p ≤₀ r ∧ q ≤₀ r :=
  ⟨fun h =>
    ⟨OneOneReducible.disjoin_left.to_many_one.trans h,
      OneOneReducible.disjoin_right.to_many_one.trans h⟩,
    fun ⟨h₁, h₂⟩ => disjoin_manyOneReducible h₁ h₂⟩


/-- Computable and injective mapping of predicates to sets of natural numbers.
-/
def toNat (p : Set α) : Set ℕ :=
  { n | p ((Encodable.decode (α := α) n).getD default) }


@[simp]
theorem toNat_manyOneReducible {p : Set α} : toNat p ≤₀ p :=
  ⟨fun n => (Encodable.decode (α := α) n).getD default,
    Computable.option_getD Computable.decode (Computable.const _), fun _ => Iff.rfl⟩


@[simp]
theorem manyOneReducible_toNat {p : Set α} : p ≤₀ toNat p :=
                                           /-
                                             α : Type u
                                             inst✝¹ : Primcodable α
                                             inst✝ : Inhabited α
                                             p : Set α
                                             ⊢ ∀ (a : α), Iff (p a) (toNat p (Encodable.encode a))
                                           -/
  ⟨Encodable.encode, Computable.encode, by simp [toNat, setOf]⟩
                                           /-
                                             🎉 no goals
                                           -/


@[simp]
theorem manyOneReducible_toNat_toNat {p : Set α} {q : Set β} : toNat p ≤₀ toNat q ↔ p ≤₀ q :=
  ⟨fun h => manyOneReducible_toNat.trans (h.trans toNat_manyOneReducible), fun h =>
    toNat_manyOneReducible.trans (h.trans manyOneReducible_toNat)⟩


@[simp]
                                                                        /-
                                                                          α : Type u
                                                                          inst✝¹ : Primcodable α
                                                                          inst✝ : Inhabited α
                                                                          p : Set α
                                                                          ⊢ ManyOneEquiv (toNat p) p
                                                                        -/
theorem toNat_manyOneEquiv {p : Set α} : ManyOneEquiv (toNat p) p := by simp [ManyOneEquiv]
                                                                        /-
                                                                          🎉 no goals
                                                                        -/


@[simp]
theorem manyOneEquiv_toNat (p : Set α) (q : Set β) :
                                                              /-
                                                                α : Type u
                                                                inst✝³ : Primcodable α
                                                                inst✝² : Inhabited α
                                                                β : Type v
                                                                inst✝¹ : Primcodable β
                                                                inst✝ : Inhabited β
                                                                p : Set α
                                                                q : Set β
                                                                ⊢ Iff (ManyOneEquiv (toNat p) (toNat q)) (ManyOneEquiv p q)
                                                              -/
    ManyOneEquiv (toNat p) (toNat q) ↔ ManyOneEquiv p q := by simp [ManyOneEquiv]
                                                              /-
                                                                🎉 no goals
                                                              -/


/-- A many-one degree is an equivalence class of sets up to many-one equivalence. -/
def ManyOneDegree : Type :=
  Quotient (⟨ManyOneEquiv, equivalence_of_manyOneEquiv⟩ : Setoid (Set ℕ))


/-- The many-one degree of a set on a primcodable type. -/
def of (p : α → Prop) : ManyOneDegree :=
  Quotient.mk'' (toNat p)


@[elab_as_elim]
protected theorem ind_on {C : ManyOneDegree → Prop} (d : ManyOneDegree)
    (h : ∀ p : Set ℕ, C (of p)) : C d :=
  Quotient.inductionOn' d h


/-- Lifts a function on sets of natural numbers to many-one degrees. -/
protected abbrev liftOn {φ} (d : ManyOneDegree) (f : Set ℕ → φ)
    (h : ∀ p q, ManyOneEquiv p q → f p = f q) : φ :=
  Quotient.liftOn' d f h


@[simp]
protected theorem liftOn_eq {φ} (p : Set ℕ) (f : Set ℕ → φ)
    (h : ∀ p q, ManyOneEquiv p q → f p = f q) : (of p).liftOn f h = f p :=
  rfl


/-- Lifts a binary function on sets of natural numbers to many-one degrees. -/
@[reducible, simp]
protected def liftOn₂ {φ} (d₁ d₂ : ManyOneDegree) (f : Set ℕ → Set ℕ → φ)
    (h : ∀ p₁ p₂ q₁ q₂, ManyOneEquiv p₁ p₂ → ManyOneEquiv q₁ q₂ → f p₁ q₁ = f p₂ q₂) : φ :=
                                                                  /-
                                                                    α : Type u
                                                                    inst✝³ : Primcodable α
                                                                    inst✝² : Inhabited α
                                                                    β : Type v
                                                                    inst✝¹ : Primcodable β
                                                                    inst✝ : Inhabited β
                                                                    φ : Sort ?u.26141
                                                                    d₁ d₂ : ManyOneDegree
                                                                    f : Set Nat → Set Nat → φ
                                                                    h : ∀ (p₁ p₂ q₁ q₂ : Nat → Prop), ManyOneEquiv p₁ p₂ → ManyOneEquiv q₁ q₂ → Eq …
                                                                    p : Set Nat
                                                                    x✝¹ x✝ : Nat → Prop
                                                                    hq : ManyOneEquiv x✝¹ x✝
                                                                    ⊢ ManyOneEquiv p p
                                                                  -/
  d₁.liftOn (fun p => d₂.liftOn (f p) fun _ _ hq => h _ _ _ _ (by rfl) hq)
                                                                  /-
                                                                    🎉 no goals
                                                                  -/
    (by
      /-
        α : Type u
        inst✝³ : Primcodable α
        inst✝² : Inhabited α
        β : Type v
        inst✝¹ : Primcodable β
        inst✝ : Inhabited β
        φ : Sort ?u.26141
        d₁ d₂ : ManyOneDegree
        f : Set Nat → Set Nat → φ
        h : ∀ (p₁ p₂ q₁ q₂ : Nat → Prop), ManyOneEquiv p₁ p₂ → ManyOneEquiv q₁ q₂ → Eq …
        ⊢ ∀ (p q : Nat → Prop), ManyOneEquiv p q → Eq ((fun p => d₂.liftOn (f p) ⋯) p) …
      -/
      intro p₁ p₂ hp
      /-
        α : Type u
        inst✝³ : Primcodable α
        inst✝² : Inhabited α
        β : Type v
        inst✝¹ : Primcodable β
        inst✝ : Inhabited β
        φ : Sort ?u.26141
        d₁ d₂ : ManyOneDegree
        f : Set Nat → Set Nat → φ
        h : ∀ (p₁ p₂ q₁ q₂ : Nat → Prop), ManyOneEquiv p₁ p₂ → ManyOneEquiv q₁ q₂ → Eq …
        p₁ p₂ : Nat → Prop
        hp : ManyOneEquiv p₁ p₂
        ⊢ Eq ((fun p => d₂.liftOn (f p) ⋯) p₁) ((fun p => d₂.liftOn (f p) ⋯) p₂)
      -/
      induction d₂ using ManyOneDegree.ind_on
      /-
        case h
        α : Type u
        inst✝³ : Primcodable α
        inst✝² : Inhabited α
        β : Type v
        inst✝¹ : Primcodable β
        inst✝ : Inhabited β
        φ : Sort ?u.26141
        d₁ : ManyOneDegree
        f : Set Nat → Set Nat → φ
        h : ∀ (p₁ p₂ q₁ q₂ : Nat → Prop), ManyOneEquiv p₁ p₂ → ManyOneEquiv q₁ q₂ → Eq …
        p₁ p₂ : Nat → Prop
        hp : ManyOneEquiv p₁ p₂
        p✝ : Set Nat
        ⊢ Eq ((fun p => (ManyOneDegree.of p✝).liftOn (f p) ⋯) p₁) ((fun p => (ManyOneD …
      -/
      apply h
        /-
          case h.a
          α : Type u
          inst✝³ : Primcodable α
          inst✝² : Inhabited α
          β : Type v
          inst✝¹ : Primcodable β
          inst✝ : Inhabited β
          φ : Sort ?u.26141
          d₁ : ManyOneDegree
          f : Set Nat → Set Nat → φ
          h : ∀ (p₁ p₂ q₁ q₂ : Nat → Prop), ManyOneEquiv p₁ p₂ → ManyOneEquiv q₁ q₂ → Eq …
          p₁ p₂ : Nat → Prop
          hp : ManyOneEquiv p₁ p₂
          p✝ : Set Nat
          ⊢ ManyOneEquiv p₁ p₂
        -/
      · assumption
        /-
          🎉 no goals
        -/
        /-
          case h.a
          α : Type u
          inst✝³ : Primcodable α
          inst✝² : Inhabited α
          β : Type v
          inst✝¹ : Primcodable β
          inst✝ : Inhabited β
          φ : Sort ?u.26141
          d₁ : ManyOneDegree
          f : Set Nat → Set Nat → φ
          h : ∀ (p₁ p₂ q₁ q₂ : Nat → Prop), ManyOneEquiv p₁ p₂ → ManyOneEquiv q₁ q₂ → Eq …
          p₁ p₂ : Nat → Prop
          hp : ManyOneEquiv p₁ p₂
          p✝ : Set Nat
          ⊢ ManyOneEquiv (toNat p✝) (toNat p✝)
        -/
      · rfl)
        /-
          🎉 no goals
        -/


@[simp]
protected theorem liftOn₂_eq {φ} (p q : Set ℕ) (f : Set ℕ → Set ℕ → φ)
    (h : ∀ p₁ p₂ q₁ q₂, ManyOneEquiv p₁ p₂ → ManyOneEquiv q₁ q₂ → f p₁ q₁ = f p₂ q₂) :
    (of p).liftOn₂ (of q) f h = f p q :=
  rfl


@[simp]
theorem of_eq_of {p : α → Prop} {q : β → Prop} : of p = of q ↔ ManyOneEquiv p q := by
  /-
    α : Type u
    inst✝³ : Primcodable α
    inst✝² : Inhabited α
    β : Type v
    inst✝¹ : Primcodable β
    inst✝ : Inhabited β
    p : α → Prop
    q : β → Prop
    ⊢ Iff (Eq (ManyOneDegree.of p) (ManyOneDegree.of q)) (ManyOneEquiv p q)
  -/
  rw [of, of, Quotient.eq'']
  /-
    α : Type u
    inst✝³ : Primcodable α
    inst✝² : Inhabited α
    β : Type v
    inst✝¹ : Primcodable β
    inst✝ : Inhabited β
    p : α → Prop
    q : β → Prop
    ⊢ Iff ({ r := ManyOneEquiv, iseqv := ManyOneDegree.proof_1 } (toNat p) (toNat  …
  -/
  simp
  /-
    🎉 no goals
  -/


instance instInhabited : Inhabited ManyOneDegree :=
  ⟨of (∅ : Set ℕ)⟩


/-- For many-one degrees `d₁` and `d₂`, `d₁ ≤ d₂` if the sets in `d₁` are many-one reducible to the
sets in `d₂`.
-/
instance instLE : LE ManyOneDegree :=
  ⟨fun d₁ d₂ =>
    ManyOneDegree.liftOn₂ d₁ d₂ (· ≤₀ ·) fun _p₁ _p₂ _q₁ _q₂ hp hq =>
      propext (hp.le_congr_left.trans hq.le_congr_right)⟩


@[simp]
theorem of_le_of {p : α → Prop} {q : β → Prop} : of p ≤ of q ↔ p ≤₀ q :=
  manyOneReducible_toNat_toNat


private theorem le_refl (d : ManyOneDegree) : d ≤ d := by
  /-
    d : ManyOneDegree
    ⊢ LE.le d d
  -/
  induction d using ManyOneDegree.ind_on; simp; rfl
                                                /-
                                                  🎉 no goals
                                                -/


private theorem le_antisymm {d₁ d₂ : ManyOneDegree} : d₁ ≤ d₂ → d₂ ≤ d₁ → d₁ = d₂ := by
  /-
    d₁ d₂ : ManyOneDegree
    ⊢ LE.le d₁ d₂ → LE.le d₂ d₁ → Eq d₁ d₂
  -/
  induction d₁ using ManyOneDegree.ind_on
  /-
    case h
    d₂ : ManyOneDegree
    p✝ : Set Nat
    ⊢ LE.le (ManyOneDegree.of p✝) d₂ → LE.le d₂ (ManyOneDegree.of p✝) → Eq (ManyOn …
  -/
  induction d₂ using ManyOneDegree.ind_on
  /-
    case h.h
    p✝¹ p✝ : Set Nat
    ⊢ LE.le (ManyOneDegree.of p✝¹) (ManyOneDegree.of p✝) → LE.le (ManyOneDegree.of …
  -/
  intro hp hq
  /-
    case h.h
    p✝¹ p✝ : Set Nat
    hp : LE.le (ManyOneDegree.of p✝¹) (ManyOneDegree.of p✝)
    hq : LE.le (ManyOneDegree.of p✝) (ManyOneDegree.of p✝¹)
    ⊢ Eq (ManyOneDegree.of p✝¹) (ManyOneDegree.of p✝)
  -/
  simp_all only [ManyOneEquiv, of_le_of, of_eq_of, true_and]
  /-
    🎉 no goals
  -/


private theorem le_trans {d₁ d₂ d₃ : ManyOneDegree} : d₁ ≤ d₂ → d₂ ≤ d₃ → d₁ ≤ d₃ := by
  /-
    d₁ d₂ d₃ : ManyOneDegree
    ⊢ LE.le d₁ d₂ → LE.le d₂ d₃ → LE.le d₁ d₃
  -/
  induction d₁ using ManyOneDegree.ind_on
  /-
    case h
    d₂ d₃ : ManyOneDegree
    p✝ : Set Nat
    ⊢ LE.le (ManyOneDegree.of p✝) d₂ → LE.le d₂ d₃ → LE.le (ManyOneDegree.of p✝) d₃
  -/
  induction d₂ using ManyOneDegree.ind_on
  /-
    case h.h
    d₃ : ManyOneDegree
    p✝¹ p✝ : Set Nat
    ⊢ LE.le (ManyOneDegree.of p✝¹) (ManyOneDegree.of p✝) → LE.le (ManyOneDegree.of …
  -/
  induction d₃ using ManyOneDegree.ind_on
  /-
    case h.h.h
    p✝² p✝¹ p✝ : Set Nat
    ⊢ LE.le (ManyOneDegree.of p✝²) (ManyOneDegree.of p✝¹) → LE.le (ManyOneDegree.o …
  -/
  apply ManyOneReducible.trans
  /-
    🎉 no goals
  -/


instance instPartialOrder : PartialOrder ManyOneDegree where
  le := (· ≤ ·)
  le_refl := le_refl
  le_trans _ _ _ := le_trans
  le_antisymm _ _ := le_antisymm


/-- The join of two degrees, induced by the disjoint union of two underlying sets. -/
instance instAdd : Add ManyOneDegree :=
  ⟨fun d₁ d₂ =>
    d₁.liftOn₂ d₂ (fun a b => of (a ⊕' b))
      (by
        /-
          α : Type u
          inst✝³ : Primcodable α
          inst✝² : Inhabited α
          β : Type v
          inst✝¹ : Primcodable β
          inst✝ : Inhabited β
          d₁ d₂ : ManyOneDegree
          ⊢ ∀ (p₁ p₂ q₁ q₂ : Nat → Prop), ManyOneEquiv p₁ p₂ → ManyOneEquiv q₁ q₂ → Eq ( …
        -/
        rintro a b c d ⟨hl₁, hr₁⟩ ⟨hl₂, hr₂⟩
        /-
          case intro.intro
          α : Type u
          inst✝³ : Primcodable α
          inst✝² : Inhabited α
          β : Type v
          inst✝¹ : Primcodable β
          inst✝ : Inhabited β
          d₁ d₂ : ManyOneDegree
          a b c d : Nat → Prop
          hl₁ : ManyOneReducible a b
          hr₁ : ManyOneReducible b a
          hl₂ : ManyOneReducible c d
          hr₂ : ManyOneReducible d c
          ⊢ Eq ((fun a b => ManyOneDegree.of (Sum.elim a b)) a c) ((fun a b => ManyOneDe …
        -/
        rw [of_eq_of]
        exact
          ⟨disjoin_manyOneReducible (hl₁.trans OneOneReducible.disjoin_left.to_many_one)
              (hl₂.trans OneOneReducible.disjoin_right.to_many_one),
            disjoin_manyOneReducible (hr₁.trans OneOneReducible.disjoin_left.to_many_one)
              (hr₂.trans OneOneReducible.disjoin_right.to_many_one)⟩)⟩


@[simp]
theorem add_of (p : Set α) (q : Set β) : of (p ⊕' q) = of p + of q :=
  of_eq_of.mpr
    ⟨disjoin_manyOneReducible
        (manyOneReducible_toNat.trans OneOneReducible.disjoin_left.to_many_one)
        (manyOneReducible_toNat.trans OneOneReducible.disjoin_right.to_many_one),
      disjoin_manyOneReducible
        (toNat_manyOneReducible.trans OneOneReducible.disjoin_left.to_many_one)
        (toNat_manyOneReducible.trans OneOneReducible.disjoin_right.to_many_one)⟩


@[simp]
protected theorem add_le {d₁ d₂ d₃ : ManyOneDegree} : d₁ + d₂ ≤ d₃ ↔ d₁ ≤ d₃ ∧ d₂ ≤ d₃ := by
  /-
    d₁ d₂ d₃ : ManyOneDegree
    ⊢ Iff (LE.le (HAdd.hAdd d₁ d₂) d₃) (And (LE.le d₁ d₃) (LE.le d₂ d₃))
  -/
  induction d₁ using ManyOneDegree.ind_on
  /-
    case h
    d₂ d₃ : ManyOneDegree
    p✝ : Set Nat
    ⊢ Iff (LE.le (HAdd.hAdd (ManyOneDegree.of p✝) d₂) d₃) (And (LE.le (ManyOneDegr …
  -/
  induction d₂ using ManyOneDegree.ind_on
  /-
    case h.h
    d₃ : ManyOneDegree
    p✝¹ p✝ : Set Nat
    ⊢ Iff (LE.le (HAdd.hAdd (ManyOneDegree.of p✝¹) (ManyOneDegree.of p✝)) d₃) (And …
  -/
  induction d₃ using ManyOneDegree.ind_on
  /-
    case h.h.h
    p✝² p✝¹ p✝ : Set Nat
    ⊢ Iff (LE.le (HAdd.hAdd (ManyOneDegree.of p✝²) (ManyOneDegree.of p✝¹)) (ManyOn …
  -/
  simpa only [← add_of, of_le_of] using disjoin_le
  /-
    🎉 no goals
  -/


@[simp]
protected theorem le_add_left (d₁ d₂ : ManyOneDegree) : d₁ ≤ d₁ + d₂ :=
  (ManyOneDegree.add_le.1 (le_refl _)).1


@[simp]
protected theorem le_add_right (d₁ d₂ : ManyOneDegree) : d₂ ≤ d₁ + d₂ :=
  (ManyOneDegree.add_le.1 (le_refl _)).2


instance instSemilatticeSup : SemilatticeSup ManyOneDegree :=
  { ManyOneDegree.instPartialOrder with
    sup := (· + ·)
    le_sup_left := ManyOneDegree.le_add_left
    le_sup_right := ManyOneDegree.le_add_right
    sup_le := fun _ _ _ h₁ h₂ => ManyOneDegree.add_le.2 ⟨h₁, h₂⟩ }


