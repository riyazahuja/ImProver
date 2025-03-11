universe uT uN in
/-- Rule that rewrites a single nonterminal to any string (a list of symbols). -/
@[ext]
structure ContextFreeRule (T : Type uT) (N : Type uN) where
  /-- Input nonterminal a.k.a. left-hand side. -/
  input : N
  /-- Output string a.k.a. right-hand side. -/
  output : List (Symbol T N)


/-- Context-free grammar that generates words over the alphabet `T` (a type of terminals). -/
structure ContextFreeGrammar.{uN,uT} (T : Type uT) where
  /-- Type of nonterminals. -/
  NT : Type uN
  /-- Initial nonterminal. -/
  initial : NT
  /-- Rewrite rules. -/
  rules : Finset (ContextFreeRule T NT)


/-- Inductive definition of a single application of a given context-free rule `r` to a string `u`;
`r.Rewrites u v` means that the `r` sends `u` to `v` (there may be multiple such strings `v`). -/
inductive Rewrites (r : ContextFreeRule T N) : List (Symbol T N) → List (Symbol T N) → Prop
  /-- The replacement is at the start of the remaining string. -/
  | head (s : List (Symbol T N)) :
      r.Rewrites (Symbol.nonterminal r.input :: s) (r.output ++ s)
  /-- There is a replacement later in the string. -/
  | cons (x : Symbol T N) {s₁ s₂ : List (Symbol T N)} (hrs : Rewrites r s₁ s₂) :
      r.Rewrites (x :: s₁) (x :: s₂)


lemma Rewrites.exists_parts (hr : r.Rewrites u v) :
    ∃ p q : List (Symbol T N),
      u = p ++ [Symbol.nonterminal r.input] ++ q ∧ v = p ++ r.output ++ q := by
  induction hr with
  | head s =>
    use [], s
    simp
  | cons x _ ih =>
    rcases ih with ⟨p', q', rfl, rfl⟩
    use x :: p', q'
    simp


lemma Rewrites.input_output : r.Rewrites [.nonterminal r.input] r.output := by
  /-
    T : Type uT
    N : Type uN
    r : ContextFreeRule T N
    ⊢ r.Rewrites (List.cons (Symbol.nonterminal r.input) List.nil) r.output
  -/
  simpa using head []
  /-
    🎉 no goals
  -/


lemma rewrites_of_exists_parts (r : ContextFreeRule T N) (p q : List (Symbol T N)) :
    r.Rewrites (p ++ [Symbol.nonterminal r.input] ++ q) (p ++ r.output ++ q) := by
  induction p with
  | nil         => exact Rewrites.head q
  | cons d l ih => exact Rewrites.cons d ih


/-- Rule `r` rewrites string `u` is to string `v` iff they share both a prefix `p` and postfix `q`
such that the remaining middle part of `u` is the input of `r` and the remaining middle part
of `u` is the output of `r`. -/
theorem rewrites_iff :
    r.Rewrites u v ↔ ∃ p q : List (Symbol T N),
      u = p ++ [Symbol.nonterminal r.input] ++ q ∧ v = p ++ r.output ++ q :=
                             /-
                               T : Type uT
                               N : Type uN
                               r : ContextFreeRule T N
                               u v : List (Symbol T N)
                               ⊢ (Exists fun p => Exists fun q => And (Eq u (HAppend.hAppend (HAppend.hAppend …
                             -/
  ⟨Rewrites.exists_parts, by rintro ⟨p, q, rfl, rfl⟩; apply rewrites_of_exists_parts⟩
                                                      /-
                                                        🎉 no goals
                                                      -/


/-- Add extra prefix to context-free rewriting. -/
lemma Rewrites.append_left (hvw : r.Rewrites u v) (p : List (Symbol T N)) :
    r.Rewrites (p ++ u) (p ++ v) := by
  /-
    T : Type uT
    N : Type uN
    r : ContextFreeRule T N
    u v : List (Symbol T N)
    hvw : r.Rewrites u v
    p : List (Symbol T N)
    ⊢ r.Rewrites (HAppend.hAppend p u) (HAppend.hAppend p v)
  -/
  rw [rewrites_iff] at *
  /-
    T : Type uT
    N : Type uN
    r : ContextFreeRule T N
    u v : List (Symbol T N)
    hvw : Exists fun p => Exists fun q => And (Eq u (HAppend.hAppend (HAppend.hApp …
    p : List (Symbol T N)
    ⊢ Exists fun p_1 => Exists fun q => And (Eq (HAppend.hAppend p u) (HAppend.hAp …
  -/
  rcases hvw with ⟨x, y, hxy⟩
  /-
    case intro.intro
    T : Type uT
    N : Type uN
    r : ContextFreeRule T N
    u v p x y : List (Symbol T N)
    hxy : And (Eq u (HAppend.hAppend (HAppend.hAppend x (List.cons (Symbol.nonterm …
    ⊢ Exists fun p_1 => Exists fun q => And (Eq (HAppend.hAppend p u) (HAppend.hAp …
  -/
  use p ++ x, y
  /-
    case h
    T : Type uT
    N : Type uN
    r : ContextFreeRule T N
    u v p x y : List (Symbol T N)
    hxy : And (Eq u (HAppend.hAppend (HAppend.hAppend x (List.cons (Symbol.nonterm …
    ⊢ And (Eq (HAppend.hAppend p u) (HAppend.hAppend (HAppend.hAppend (HAppend.hAp …
  -/
  simp_all
  /-
    🎉 no goals
  -/


/-- Add extra postfix to context-free rewriting. -/
lemma Rewrites.append_right (hvw : r.Rewrites u v) (p : List (Symbol T N)) :
    r.Rewrites (u ++ p) (v ++ p) := by
  /-
    T : Type uT
    N : Type uN
    r : ContextFreeRule T N
    u v : List (Symbol T N)
    hvw : r.Rewrites u v
    p : List (Symbol T N)
    ⊢ r.Rewrites (HAppend.hAppend u p) (HAppend.hAppend v p)
  -/
  rw [rewrites_iff] at *
  /-
    T : Type uT
    N : Type uN
    r : ContextFreeRule T N
    u v : List (Symbol T N)
    hvw : Exists fun p => Exists fun q => And (Eq u (HAppend.hAppend (HAppend.hApp …
    p : List (Symbol T N)
    ⊢ Exists fun p_1 => Exists fun q => And (Eq (HAppend.hAppend u p) (HAppend.hAp …
  -/
  rcases hvw with ⟨x, y, hxy⟩
  /-
    case intro.intro
    T : Type uT
    N : Type uN
    r : ContextFreeRule T N
    u v p x y : List (Symbol T N)
    hxy : And (Eq u (HAppend.hAppend (HAppend.hAppend x (List.cons (Symbol.nonterm …
    ⊢ Exists fun p_1 => Exists fun q => And (Eq (HAppend.hAppend u p) (HAppend.hAp …
  -/
  use x, y ++ p
  /-
    case h
    T : Type uT
    N : Type uN
    r : ContextFreeRule T N
    u v p x y : List (Symbol T N)
    hxy : And (Eq u (HAppend.hAppend (HAppend.hAppend x (List.cons (Symbol.nonterm …
    ⊢ And (Eq (HAppend.hAppend u p) (HAppend.hAppend (HAppend.hAppend x (List.cons …
  -/
  simp_all
  /-
    🎉 no goals
  -/


/-- Given a context-free grammar `g` and strings `u` and `v`
`g.Produces u v` means that one step of a context-free transformation by a rule from `g` sends
`u` to `v`. -/
def Produces (g : ContextFreeGrammar.{uN} T) (u v : List (Symbol T g.NT)) : Prop :=
  ∃ r ∈ g.rules, r.Rewrites u v


/-- Given a context-free grammar `g` and strings `u` and `v`
`g.Derives u v` means that `g` can transform `u` to `v` in some number of rewriting steps. -/
abbrev Derives (g : ContextFreeGrammar.{uN} T) :
    List (Symbol T g.NT) → List (Symbol T g.NT) → Prop :=
  Relation.ReflTransGen g.Produces


/-- Given a context-free grammar `g` and a string `s`
`g.Generates s` means that `g` can transform its initial nonterminal to `s` in some number of
rewriting steps. -/
def Generates (g : ContextFreeGrammar.{uN} T) (s : List (Symbol T g.NT)) : Prop :=
  g.Derives [Symbol.nonterminal g.initial] s


/-- The language (set of words) that can be generated by a given context-free grammar `g`. -/
def language (g : ContextFreeGrammar.{uN} T) : Language T :=
  { w | g.Generates (List.map Symbol.terminal w) }


/-- A given word `w` belongs to the language generated by a given context-free grammar `g` iff
`g` can derive the word `w` (wrapped as a string) from the initial nonterminal of `g` in some
number of steps. -/
@[simp]
lemma mem_language_iff (g : ContextFreeGrammar.{uN} T) (w : List T) :
    w ∈ g.language ↔ g.Derives [Symbol.nonterminal g.initial] (List.map Symbol.terminal w) := by
  /-
    T : Type uT
    g : ContextFreeGrammar T
    w : List T
    ⊢ Iff (Membership.mem g.language w) (g.Derives (List.cons (Symbol.nonterminal  …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[refl]
lemma Derives.refl (w : List (Symbol T g.NT)) : g.Derives w w :=
  Relation.ReflTransGen.refl


lemma Produces.single {v w : List (Symbol T g.NT)} (hvw : g.Produces v w) : g.Derives v w :=
  Relation.ReflTransGen.single hvw


@[trans]
lemma Derives.trans {u v w : List (Symbol T g.NT)} (huv : g.Derives u v) (hvw : g.Derives v w) :
    g.Derives u w :=
  Relation.ReflTransGen.trans huv hvw


lemma Derives.trans_produces {u v w : List (Symbol T g.NT)}
    (huv : g.Derives u v) (hvw : g.Produces v w) :
    g.Derives u w :=
  huv.trans hvw.single


lemma Produces.trans_derives {u v w : List (Symbol T g.NT)}
    (huv : g.Produces u v) (hvw : g.Derives v w) :
    g.Derives u w :=
  huv.single.trans hvw


lemma Derives.eq_or_head {u w : List (Symbol T g.NT)} (huw : g.Derives u w) :
    u = w ∨ ∃ v : List (Symbol T g.NT), g.Produces u v ∧ g.Derives v w :=
  Relation.ReflTransGen.cases_head huw


lemma Derives.eq_or_tail {u w : List (Symbol T g.NT)} (huw : g.Derives u w) :
    u = w ∨ ∃ v : List (Symbol T g.NT), g.Derives u v ∧ g.Produces v w :=
  (Relation.ReflTransGen.cases_tail huw).casesOn (Or.inl ∘ Eq.symm) Or.inr


/-- Add extra prefix to context-free producing. -/
lemma Produces.append_left {v w : List (Symbol T g.NT)}
    (hvw : g.Produces v w) (p : List (Symbol T g.NT)) :
    g.Produces (p ++ v) (p ++ w) :=
  match hvw with | ⟨r, hrmem, hrvw⟩ => ⟨r, hrmem, hrvw.append_left p⟩


/-- Add extra postfix to context-free producing. -/
lemma Produces.append_right {v w : List (Symbol T g.NT)}
    (hvw : g.Produces v w) (p : List (Symbol T g.NT)) :
    g.Produces (v ++ p) (w ++ p) :=
  match hvw with | ⟨r, hrmem, hrvw⟩ => ⟨r, hrmem, hrvw.append_right p⟩


/-- Add extra prefix to context-free deriving. -/
lemma Derives.append_left {v w : List (Symbol T g.NT)}
    (hvw : g.Derives v w) (p : List (Symbol T g.NT)) :
    g.Derives (p ++ v) (p ++ w) := by
  induction hvw with
  | refl => rfl
  | tail _ last ih => exact ih.trans_produces <| last.append_left p


/-- Add extra postfix to context-free deriving. -/
lemma Derives.append_right {v w : List (Symbol T g.NT)}
    (hvw : g.Derives v w) (p : List (Symbol T g.NT)) :
    g.Derives (v ++ p) (w ++ p) := by
  induction hvw with
  | refl => rfl
  | tail _ last ih => exact ih.trans_produces <| last.append_right p


/-- Context-free languages are defined by context-free grammars. -/
def Language.IsContextFree (L : Language T) : Prop :=
  ∃ g : ContextFreeGrammar.{0} T, g.language = L


/-- Rules for a grammar for a reversed language. -/
def reverse (r : ContextFreeRule T N) : ContextFreeRule T N := ⟨r.input, r.output.reverse⟩


                                                                                      /-
                                                                                        T : Type uT
                                                                                        N : Type uN
                                                                                        r : ContextFreeRule T N
                                                                                        ⊢ Eq r.reverse.reverse r
                                                                                      -/
@[simp] lemma reverse_reverse (r : ContextFreeRule T N) : r.reverse.reverse = r := by simp [reverse]
                                                                                      /-
                                                                                        🎉 no goals
                                                                                      -/


@[simp] lemma reverse_comp_reverse :
                                                                               /-
                                                                                 T : Type uT
                                                                                 N : Type uN
                                                                                 ⊢ Eq (Function.comp ContextFreeRule.reverse ContextFreeRule.reverse) id
                                                                               -/
    reverse ∘ reverse = (id : ContextFreeRule T N → ContextFreeRule T N) := by ext : 1; simp
                                                                                        /-
                                                                                          🎉 no goals
                                                                                        -/


lemma reverse_involutive : Involutive (reverse : ContextFreeRule T N → ContextFreeRule T N) :=
  reverse_reverse


lemma reverse_bijective : Bijective (reverse : ContextFreeRule T N → ContextFreeRule T N) :=
  reverse_involutive.bijective


lemma reverse_injective : Injective (reverse : ContextFreeRule T N → ContextFreeRule T N) :=
  reverse_involutive.injective


lemma reverse_surjective : Surjective (reverse : ContextFreeRule T N → ContextFreeRule T N) :=
  reverse_involutive.surjective


protected lemma Rewrites.reverse : ∀ {u v}, r.Rewrites u v → r.reverse.Rewrites u.reverse v.reverse
                       /-
                         T : Type uT
                         N : Type uN
                         r : ContextFreeRule T N
                         s : List (Symbol T N)
                         ⊢ r.reverse.Rewrites (List.cons (Symbol.nonterminal r.input) s).reverse (HAppe …
                       -/
  | _, _, head s => by simpa using .append_left .input_output _
                       /-
                         🎉 no goals
                       -/
                                    /-
                                      T : Type uT
                                      N : Type uN
                                      r : ContextFreeRule T N
                                      x : Symbol T N
                                      u v : List (Symbol T N)
                                      h : r.Rewrites u v
                                      ⊢ r.reverse.Rewrites (List.cons x u).reverse (List.cons x v).reverse
                                    -/
  | _, _, @cons _ _ _ x u v h => by simpa using h.reverse.append_right _
                                    /-
                                      🎉 no goals
                                    -/


lemma rewrites_reverse : r.reverse.Rewrites u.reverse v.reverse ↔ r.Rewrites u v :=
              /-
                T : Type uT
                N : Type uN
                r : ContextFreeRule T N
                u v : List (Symbol T N)
                h : r.reverse.Rewrites u.reverse v.reverse
                ⊢ r.Rewrites u v
              -/
  ⟨fun h ↦ by simpa using h.reverse, .reverse⟩
              /-
                🎉 no goals
              -/


@[simp] lemma rewrites_reverse_comm : r.reverse.Rewrites u v ↔ r.Rewrites u.reverse v.reverse := by
  /-
    T : Type uT
    N : Type uN
    r : ContextFreeRule T N
    u v : List (Symbol T N)
    ⊢ Iff (r.reverse.Rewrites u v) (r.Rewrites u.reverse v.reverse)
  -/
  rw [← rewrites_reverse, reverse_reverse]
  /-
    🎉 no goals
  -/


/-- Grammar for a reversed language. -/
@[simps] def reverse (g : ContextFreeGrammar T) : ContextFreeGrammar T :=
  ⟨g.NT, g.initial, g.rules.map (⟨ContextFreeRule.reverse, ContextFreeRule.reverse_injective⟩)⟩


@[simp] lemma reverse_reverse (g : ContextFreeGrammar T) : g.reverse.reverse = g := by
  /-
    T : Type uT
    g : ContextFreeGrammar T
    ⊢ Eq g.reverse.reverse g
  -/
  simp [reverse, Finset.map_map]
  /-
    🎉 no goals
  -/


lemma reverse_involutive : Involutive (reverse : ContextFreeGrammar T → ContextFreeGrammar T) :=
  reverse_reverse


lemma reverse_bijective : Bijective (reverse : ContextFreeGrammar T → ContextFreeGrammar T) :=
  reverse_involutive.bijective


lemma reverse_injective : Injective (reverse : ContextFreeGrammar T → ContextFreeGrammar T) :=
  reverse_involutive.injective


lemma reverse_surjective : Surjective (reverse : ContextFreeGrammar T → ContextFreeGrammar T) :=
  reverse_involutive.surjective


lemma produces_reverse : g.reverse.Produces u.reverse v.reverse ↔ g.Produces u v :=
  (Equiv.ofBijective _ ContextFreeRule.reverse_bijective).exists_congr
        /-
          T : Type uT
          g : ContextFreeGrammar T
          u v : List (Symbol T g.NT)
          ⊢ ∀ (a : ContextFreeRule T g.reverse.NT), Iff (And (Membership.mem g.reverse.r …
        -/
    (by simp [ContextFreeRule.reverse_involutive.eq_iff])
        /-
          🎉 no goals
        -/


alias ⟨_, Produces.reverse⟩ := produces_reverse


@[simp] lemma produces_reverse_comm : g.reverse.Produces u v ↔ g.Produces u.reverse v.reverse :=
  (Equiv.ofBijective _ ContextFreeRule.reverse_bijective).exists_congr
        /-
          T : Type uT
          g : ContextFreeGrammar T
          u v : List (Symbol T g.NT)
          ⊢ ∀ (a : ContextFreeRule T g.reverse.NT), Iff (And (Membership.mem g.reverse.r …
        -/
    (by simp [ContextFreeRule.reverse_involutive.eq_iff])
        /-
          🎉 no goals
        -/


protected lemma Derives.reverse (hg : g.Derives u v) : g.reverse.Derives u.reverse v.reverse := by
  induction hg with
  | refl => rfl
  | tail _ orig ih => exact ih.trans_produces orig.reverse


lemma derives_reverse : g.reverse.Derives u.reverse v.reverse ↔ g.Derives u v :=
              /-
                T : Type uT
                g : ContextFreeGrammar T
                u v : List (Symbol T g.NT)
                h : g.reverse.Derives u.reverse v.reverse
                ⊢ g.Derives u v
              -/
                                    /-
                                      🎉 no goals
                                    -/
                                    /-
                                      🎉 no goals
                                    -/
  ⟨fun h ↦ by convert h.reverse <;> simp, .reverse⟩
                                    /-
                                      🎉 no goals
                                    -/


@[simp] lemma derives_reverse_comm : g.reverse.Derives u v ↔ g.Derives u.reverse v.reverse := by
  /-
    T : Type uT
    g : ContextFreeGrammar T
    u v : List (Symbol T g.NT)
    ⊢ Iff (g.reverse.Derives u v) (g.Derives u.reverse v.reverse)
  -/
  rw [iff_comm, ← derives_reverse, List.reverse_reverse, List.reverse_reverse]
  /-
    🎉 no goals
  -/


                                                                              /-
                                                                                T : Type uT
                                                                                g : ContextFreeGrammar T
                                                                                u : List (Symbol T g.NT)
                                                                                ⊢ Iff (g.reverse.Generates u.reverse) (g.Generates u)
                                                                              -/
lemma generates_reverse : g.reverse.Generates u.reverse ↔ g.Generates u := by simp [Generates]
                                                                              /-
                                                                                🎉 no goals
                                                                              -/


alias ⟨_, Generates.reverse⟩ := generates_reverse


@[simp] lemma generates_reverse_comm : g.reverse.Generates u ↔ g.Generates u.reverse := by
  /-
    T : Type uT
    g : ContextFreeGrammar T
    u : List (Symbol T g.NT)
    ⊢ Iff (g.reverse.Generates u) (g.Generates u.reverse)
  -/
  simp [Generates]
  /-
    🎉 no goals
  -/


                                                                               /-
                                                                                 T : Type uT
                                                                                 g : ContextFreeGrammar T
                                                                                 ⊢ Eq g.reverse.language g.language.reverse
                                                                               -/
@[simp] lemma language_reverse : g.reverse.language = g.language.reverse := by ext; simp
                                                                                    /-
                                                                                      🎉 no goals
                                                                                    -/


/-- The class of context-free languages is closed under reversal. -/
theorem Language.IsContextFree.reverse (L : Language T) :
                                                    /-
                                                      T : Type uT
                                                      L : Language T
                                                      ⊢ L.IsContextFree → L.reverse.IsContextFree
                                                    -/
    L.IsContextFree → L.reverse.IsContextFree := by rintro ⟨g, rfl⟩; exact ⟨g.reverse, by simp⟩
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


