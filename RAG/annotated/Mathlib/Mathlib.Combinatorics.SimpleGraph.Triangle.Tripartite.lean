/-- The underlying relation of the tripartite-from-triangles graph.

Two vertices are related iff there exists a triangle index containing them both. -/
@[mk_iff] inductive Rel (t : Finset (α × β × γ)) : α ⊕ β ⊕ γ → α ⊕ β ⊕ γ → Prop
| in₀₁ ⦃a b c⦄ : (a, b, c) ∈ t → Rel t (in₀ a) (in₁ b)
| in₁₀ ⦃a b c⦄ : (a, b, c) ∈ t → Rel t (in₁ b) (in₀ a)
| in₀₂ ⦃a b c⦄ : (a, b, c) ∈ t → Rel t (in₀ a) (in₂ c)
| in₂₀ ⦃a b c⦄ : (a, b, c) ∈ t → Rel t (in₂ c) (in₀ a)
| in₁₂ ⦃a b c⦄ : (a, b, c) ∈ t → Rel t (in₁ b) (in₂ c)
| in₂₁ ⦃a b c⦄ : (a, b, c) ∈ t → Rel t (in₂ c) (in₁ b)


lemma rel_irrefl : ∀ x, ¬ Rel t x x := fun _x hx ↦ nomatch hx

                                                      /-
                                                        α : Type u_1
                                                        β : Type u_2
                                                        γ : Type u_3
                                                        t : Finset (Prod α (Prod β γ))
                                                        x y : Sum α (Sum β γ)
                                                        h : SimpleGraph.TripartiteFromTriangles.Rel t x y
                                                        ⊢ SimpleGraph.TripartiteFromTriangles.Rel t y x
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
                                                                                  /-
                                                                                    🎉 no goals
                                                                                  -/
lemma rel_symm : Symmetric (Rel t) := fun x y h ↦  by cases h <;> constructor <;> assumption
                                                                                  /-
                                                                                    🎉 no goals
                                                                                  -/


/-- The tripartite-from-triangles graph. Two vertices are related iff there exists a triangle index
containing them both. -/
def graph (t : Finset (α × β × γ)) : SimpleGraph (α ⊕ β ⊕ γ) := ⟨Rel t, rel_symm, rel_irrefl⟩


@[simp] lemma not_in₀₀ : ¬ (graph t).Adj (in₀ a) (in₀ a') := fun h ↦ nomatch h

@[simp] lemma not_in₁₁ : ¬ (graph t).Adj (in₁ b) (in₁ b') := fun h ↦ nomatch h

@[simp] lemma not_in₂₂ : ¬ (graph t).Adj (in₂ c) (in₂ c') := fun h ↦ nomatch h


@[simp] lemma in₀₁_iff : (graph t).Adj (in₀ a) (in₁ b) ↔ ∃ c, (a, b, c) ∈ t :=
      /-
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        t : Finset (Prod α (Prod β γ))
        a : α
        b : β
        ⊢ (SimpleGraph.TripartiteFromTriangles.graph t).Adj (Sum3.in₀ a) (Sum3.in₁ b)  …
      -/
  ⟨by rintro ⟨⟩; exact ⟨_, ‹_›⟩, fun ⟨_, h⟩ ↦ in₀₁ h⟩
                 /-
                   🎉 no goals
                 -/

@[simp] lemma in₁₀_iff : (graph t).Adj (in₁ b) (in₀ a) ↔ ∃ c, (a, b, c) ∈ t :=
      /-
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        t : Finset (Prod α (Prod β γ))
        a : α
        b : β
        ⊢ (SimpleGraph.TripartiteFromTriangles.graph t).Adj (Sum3.in₁ b) (Sum3.in₀ a)  …
      -/
  ⟨by rintro ⟨⟩; exact ⟨_, ‹_›⟩, fun ⟨_, h⟩ ↦ in₁₀ h⟩
                 /-
                   🎉 no goals
                 -/

@[simp] lemma in₀₂_iff : (graph t).Adj (in₀ a) (in₂ c) ↔ ∃ b, (a, b, c) ∈ t :=
      /-
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        t : Finset (Prod α (Prod β γ))
        a : α
        c : γ
        ⊢ (SimpleGraph.TripartiteFromTriangles.graph t).Adj (Sum3.in₀ a) (Sum3.in₂ c)  …
      -/
  ⟨by rintro ⟨⟩; exact ⟨_, ‹_›⟩, fun ⟨_, h⟩ ↦ in₀₂ h⟩
                 /-
                   🎉 no goals
                 -/

@[simp] lemma in₂₀_iff : (graph t).Adj (in₂ c) (in₀ a) ↔ ∃ b, (a, b, c) ∈ t :=
      /-
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        t : Finset (Prod α (Prod β γ))
        a : α
        c : γ
        ⊢ (SimpleGraph.TripartiteFromTriangles.graph t).Adj (Sum3.in₂ c) (Sum3.in₀ a)  …
      -/
  ⟨by rintro ⟨⟩; exact ⟨_, ‹_›⟩, fun ⟨_, h⟩ ↦ in₂₀ h⟩
                 /-
                   🎉 no goals
                 -/

@[simp] lemma in₁₂_iff : (graph t).Adj (in₁ b) (in₂ c) ↔ ∃ a, (a, b, c) ∈ t :=
      /-
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        t : Finset (Prod α (Prod β γ))
        b : β
        c : γ
        ⊢ (SimpleGraph.TripartiteFromTriangles.graph t).Adj (Sum3.in₁ b) (Sum3.in₂ c)  …
      -/
  ⟨by rintro ⟨⟩; exact ⟨_, ‹_›⟩, fun ⟨_, h⟩ ↦ in₁₂ h⟩
                 /-
                   🎉 no goals
                 -/

@[simp] lemma in₂₁_iff : (graph t).Adj (in₂ c) (in₁ b) ↔ ∃ a, (a, b, c) ∈ t :=
      /-
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        t : Finset (Prod α (Prod β γ))
        b : β
        c : γ
        ⊢ (SimpleGraph.TripartiteFromTriangles.graph t).Adj (Sum3.in₂ c) (Sum3.in₁ b)  …
      -/
  ⟨by rintro ⟨⟩; exact ⟨_, ‹_›⟩, fun ⟨_, h⟩ ↦ in₂₁ h⟩
                 /-
                   🎉 no goals
                 -/


lemma in₀₁_iff' :
    (graph t).Adj (in₀ a) (in₁ b) ↔ ∃ x : α × β × γ, x ∈ t ∧ x.1 = a ∧ x.2.1 = b where
           /-
             α : Type u_1
             β : Type u_2
             γ : Type u_3
             t : Finset (Prod α (Prod β γ))
             a : α
             b : β
             ⊢ (SimpleGraph.TripartiteFromTriangles.graph t).Adj (Sum3.in₀ a) (Sum3.in₁ b)  …
           -/
  mp := by rintro ⟨⟩; exact ⟨_, ‹_›, by simp⟩
                      /-
                        🎉 no goals
                      -/
            /-
              α : Type u_1
              β : Type u_2
              γ : Type u_3
              t : Finset (Prod α (Prod β γ))
              a : α
              b : β
              ⊢ (Exists fun x => And (Membership.mem t x) (And (Eq x.1 a) (Eq x.2.1 b))) → ( …
            -/
  mpr := by rintro ⟨⟨a, b, c⟩, h, rfl, rfl⟩; constructor; assumption
                                                          /-
                                                            🎉 no goals
                                                          -/

lemma in₁₀_iff' :
    (graph t).Adj (in₁ b) (in₀ a) ↔ ∃ x : α × β × γ, x ∈ t ∧ x.2.1 = b ∧ x.1 = a where
           /-
             α : Type u_1
             β : Type u_2
             γ : Type u_3
             t : Finset (Prod α (Prod β γ))
             a : α
             b : β
             ⊢ (SimpleGraph.TripartiteFromTriangles.graph t).Adj (Sum3.in₁ b) (Sum3.in₀ a)  …
           -/
  mp := by rintro ⟨⟩; exact ⟨_, ‹_›, by simp⟩
                      /-
                        🎉 no goals
                      -/
            /-
              α : Type u_1
              β : Type u_2
              γ : Type u_3
              t : Finset (Prod α (Prod β γ))
              a : α
              b : β
              ⊢ (Exists fun x => And (Membership.mem t x) (And (Eq x.2.1 b) (Eq x.1 a))) → ( …
            -/
  mpr := by rintro ⟨⟨a, b, c⟩, h, rfl, rfl⟩; constructor; assumption
                                                          /-
                                                            🎉 no goals
                                                          -/

lemma in₀₂_iff' :
    (graph t).Adj (in₀ a) (in₂ c) ↔ ∃ x : α × β × γ, x ∈ t ∧ x.1 = a ∧ x.2.2 = c where
           /-
             α : Type u_1
             β : Type u_2
             γ : Type u_3
             t : Finset (Prod α (Prod β γ))
             a : α
             c : γ
             ⊢ (SimpleGraph.TripartiteFromTriangles.graph t).Adj (Sum3.in₀ a) (Sum3.in₂ c)  …
           -/
  mp := by rintro ⟨⟩; exact ⟨_, ‹_›, by simp⟩
                      /-
                        🎉 no goals
                      -/
            /-
              α : Type u_1
              β : Type u_2
              γ : Type u_3
              t : Finset (Prod α (Prod β γ))
              a : α
              c : γ
              ⊢ (Exists fun x => And (Membership.mem t x) (And (Eq x.1 a) (Eq x.2.2 c))) → ( …
            -/
  mpr := by rintro ⟨⟨a, b, c⟩, h, rfl, rfl⟩; constructor; assumption
                                                          /-
                                                            🎉 no goals
                                                          -/

lemma in₂₀_iff' :
    (graph t).Adj (in₂ c) (in₀ a) ↔ ∃ x : α × β × γ, x ∈ t ∧ x.2.2 = c ∧ x.1 = a where
           /-
             α : Type u_1
             β : Type u_2
             γ : Type u_3
             t : Finset (Prod α (Prod β γ))
             a : α
             c : γ
             ⊢ (SimpleGraph.TripartiteFromTriangles.graph t).Adj (Sum3.in₂ c) (Sum3.in₀ a)  …
           -/
  mp := by rintro ⟨⟩; exact ⟨_, ‹_›, by simp⟩
                      /-
                        🎉 no goals
                      -/
            /-
              α : Type u_1
              β : Type u_2
              γ : Type u_3
              t : Finset (Prod α (Prod β γ))
              a : α
              c : γ
              ⊢ (Exists fun x => And (Membership.mem t x) (And (Eq x.2.2 c) (Eq x.1 a))) → ( …
            -/
  mpr := by rintro ⟨⟨a, b, c⟩, h, rfl, rfl⟩; constructor; assumption
                                                          /-
                                                            🎉 no goals
                                                          -/

lemma in₁₂_iff' :
    (graph t).Adj (in₁ b) (in₂ c) ↔ ∃ x : α × β × γ, x ∈ t ∧ x.2.1 = b ∧ x.2.2 = c where
           /-
             α : Type u_1
             β : Type u_2
             γ : Type u_3
             t : Finset (Prod α (Prod β γ))
             b : β
             c : γ
             ⊢ (SimpleGraph.TripartiteFromTriangles.graph t).Adj (Sum3.in₁ b) (Sum3.in₂ c)  …
           -/
  mp := by rintro ⟨⟩; exact ⟨_, ‹_›, by simp⟩
                      /-
                        🎉 no goals
                      -/
            /-
              α : Type u_1
              β : Type u_2
              γ : Type u_3
              t : Finset (Prod α (Prod β γ))
              b : β
              c : γ
              ⊢ (Exists fun x => And (Membership.mem t x) (And (Eq x.2.1 b) (Eq x.2.2 c))) → …
            -/
  mpr := by rintro ⟨⟨a, b, c⟩, h, rfl, rfl⟩; constructor; assumption
                                                          /-
                                                            🎉 no goals
                                                          -/

lemma in₂₁_iff' :
    (graph t).Adj (in₂ c) (in₁ b) ↔ ∃ x : α × β × γ, x ∈ t ∧ x.2.2 = c ∧ x.2.1 = b where
           /-
             α : Type u_1
             β : Type u_2
             γ : Type u_3
             t : Finset (Prod α (Prod β γ))
             b : β
             c : γ
             ⊢ (SimpleGraph.TripartiteFromTriangles.graph t).Adj (Sum3.in₂ c) (Sum3.in₁ b)  …
           -/
  mp := by rintro ⟨⟩; exact ⟨_, ‹_›, by simp⟩
                      /-
                        🎉 no goals
                      -/
            /-
              α : Type u_1
              β : Type u_2
              γ : Type u_3
              t : Finset (Prod α (Prod β γ))
              b : β
              c : γ
              ⊢ (Exists fun x => And (Membership.mem t x) (And (Eq x.2.2 c) (Eq x.2.1 b))) → …
            -/
  mpr := by rintro ⟨⟨a, b, c⟩, h, rfl, rfl⟩; constructor; assumption
                                                          /-
                                                            🎉 no goals
                                                          -/


/-- Predicate on the triangle indices for the explicit triangles to be edge-disjoint. -/
class ExplicitDisjoint (t : Finset (α × β × γ)) : Prop where
  inj₀ : ∀ ⦃a b c a'⦄, (a, b, c) ∈ t → (a', b, c) ∈ t → a = a'
  inj₁ : ∀ ⦃a b c b'⦄, (a, b, c) ∈ t → (a, b', c) ∈ t → b = b'
  inj₂ : ∀ ⦃a b c c'⦄, (a, b, c) ∈ t → (a, b, c') ∈ t → c = c'


/-- Predicate on the triangle indices for there to be no accidental triangle.

Note that we cheat a bit, since the exact translation of this informal description would have
`(a', b', c') ∈ t` as a conclusion rather than `a = a' ∨ b = b' ∨ c = c'`. Those conditions are
equivalent when the explicit triangles are edge-disjoint (which is the case we care about). -/
class NoAccidental (t : Finset (α × β × γ)) : Prop where
  eq_or_eq_or_eq : ∀ ⦃a a' b b' c c'⦄, (a', b, c) ∈ t → (a, b', c) ∈ t → (a, b, c') ∈ t →
    a = a' ∨ b = b' ∨ c = c'


instance graph.instDecidableRelAdj : DecidableRel (graph t).Adj
  | in₀ _a, in₀ _a' => Decidable.isFalse not_in₀₀
  | in₀ _a, in₁ _b' => decidable_of_iff' _ in₀₁_iff'
  | in₀ _a, in₂ _c' => decidable_of_iff' _ in₀₂_iff'
  | in₁ _b, in₀ _a' => decidable_of_iff' _ in₁₀_iff'
  | in₁ _b, in₁ _b' => Decidable.isFalse not_in₁₁
  | in₁ _b, in₂ _b' => decidable_of_iff' _ in₁₂_iff'
  | in₂ _c, in₀ _a' => decidable_of_iff' _ in₂₀_iff'
  | in₂ _c, in₁ _b' => decidable_of_iff' _ in₂₁_iff'
  | in₂ _c, in₂ _b' => Decidable.isFalse not_in₂₂


/-- This lemma reorders the elements of a triangle in the tripartite graph. It turns a triangle
`{x, y, z}` into a triangle `{a, b, c}` where `a : α `, `b : β`, `c : γ`. -/
 lemma graph_triple ⦃x y z⦄ :
  (graph t).Adj x y → (graph t).Adj x z → (graph t).Adj y z → ∃ a b c,
    ({in₀ a, in₁ b, in₂ c} : Finset (α ⊕ β ⊕ γ)) = {x, y, z} ∧ (graph t).Adj (in₀ a) (in₁ b) ∧
      (graph t).Adj (in₀ a) (in₂ c) ∧ (graph t).Adj (in₁ b) (in₂ c) := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    t : Finset (Prod α (Prod β γ))
    inst✝² : DecidableEq α
    inst✝¹ : DecidableEq β
    inst✝ : DecidableEq γ
    x y z : Sum α (Sum β γ)
    ⊢ (SimpleGraph.TripartiteFromTriangles.graph t).Adj x y → (SimpleGraph.Tripart …
  -/
  rintro (_ | _ | _) (_ | _ | _) (_ | _ | _) <;>
    refine ⟨_, _, _, by ext; simp only [Finset.mem_insert, Finset.mem_singleton]; try tauto,
                      /-
                        case in₀₁.in₀₂.in₁₂.refine_1
                        α : Type u_1
                        β : Type u_2
                        γ : Type u_3
                        t : Finset (Prod α (Prod β γ))
                        inst✝² : DecidableEq α
                        inst✝¹ : DecidableEq β
                        inst✝ : DecidableEq γ
                        a✝⁴ : α
                        b✝¹ : β
                        c✝¹ : γ
                        a✝³ : Membership.mem t { fst := a✝⁴, snd := { fst := b✝¹, snd := c✝¹ } }
                        b✝ : β
                        c✝ : γ
                        a✝² : Membership.mem t { fst := a✝⁴, snd := { fst := b✝, snd := c✝ } }
                        a✝¹ : α
                        a✝ : Membership.mem t { fst := a✝¹, snd := { fst := b✝¹, snd := c✝ } }
                        ⊢ (SimpleGraph.TripartiteFromTriangles.graph t).Adj (Sum3.in₀ a✝⁴) (Sum3.in₁ b …
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
                                      /-
                                        🎉 no goals
                                      -/
      ?_, ?_, ?_⟩ <;> constructor <;> assumption
                                      /-
                                        🎉 no goals
                                      -/


/-- The map that turns a triangle index into an explicit triangle. -/
@[simps] def toTriangle : α × β × γ ↪ Finset (α ⊕ β ⊕ γ) where
  toFun x := {in₀ x.1, in₁ x.2.1, in₂ x.2.2}
  inj' := fun ⟨a, b, c⟩ ⟨a', b', c'⟩ ↦ by simpa only [Finset.Subset.antisymm_iff, Finset.subset_iff,
    mem_insert, mem_singleton, forall_eq_or_imp, forall_eq, Prod.mk.inj_iff, or_false, false_or,
    in₀, in₁, in₂, Sum.inl.inj_iff, Sum.inr.inj_iff, reduceCtorEq] using And.left


lemma toTriangle_is3Clique (hx : x ∈ t) : (graph t).IsNClique 3 (toTriangle x) := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    t : Finset (Prod α (Prod β γ))
    x : Prod α (Prod β γ)
    inst✝² : DecidableEq α
    inst✝¹ : DecidableEq β
    inst✝ : DecidableEq γ
    hx : Membership.mem t x
    ⊢ (SimpleGraph.TripartiteFromTriangles.graph t).IsNClique 3 (SimpleGraph.Tripa …
  -/
  simp only [toTriangle_apply, is3Clique_triple_iff, in₀₁_iff, in₀₂_iff, in₁₂_iff]
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    t : Finset (Prod α (Prod β γ))
    x : Prod α (Prod β γ)
    inst✝² : DecidableEq α
    inst✝¹ : DecidableEq β
    inst✝ : DecidableEq γ
    hx : Membership.mem t x
    ⊢ And (Exists fun c => Membership.mem t { fst := x.1, snd := { fst := x.2.1, s …
  -/
  exact ⟨⟨_, hx⟩, ⟨_, hx⟩, _, hx⟩
  /-
    🎉 no goals
  -/


lemma exists_mem_toTriangle {x y : α ⊕ β ⊕ γ} (hxy : (graph t).Adj x y) :
                                                       /-
                                                         α : Type u_1
                                                         β : Type u_2
                                                         γ : Type u_3
                                                         t : Finset (Prod α (Prod β γ))
                                                         inst✝² : DecidableEq α
                                                         inst✝¹ : DecidableEq β
                                                         inst✝ : DecidableEq γ
                                                         x y : Sum α (Sum β γ)
                                                         hxy : (SimpleGraph.TripartiteFromTriangles.graph t).Adj x y
                                                         ⊢ Exists fun z => And (Membership.mem t z) (And (Membership.mem (SimpleGraph.T …
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
                                                                     /-
                                                                       🎉 no goals
                                                                     -/
    ∃ z ∈ t, x ∈ toTriangle z ∧ y ∈ toTriangle z := by cases hxy <;> exact ⟨_, ‹_›, by simp⟩
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


nonrec lemma is3Clique_iff [NoAccidental t] {s : Finset (α ⊕ β ⊕ γ)} :
    (graph t).IsNClique 3 s ↔ ∃ x, x ∈ t ∧ toTriangle x = s := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    t : Finset (Prod α (Prod β γ))
    inst✝³ : DecidableEq α
    inst✝² : DecidableEq β
    inst✝¹ : DecidableEq γ
    inst✝ : SimpleGraph.TripartiteFromTriangles.NoAccidental t
    s : Finset (Sum α (Sum β γ))
    ⊢ Iff ((SimpleGraph.TripartiteFromTriangles.graph t).IsNClique 3 s) (Exists fu …
  -/
  refine ⟨fun h ↦ ?_, ?_⟩
    /-
      case refine_1
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      t : Finset (Prod α (Prod β γ))
      inst✝³ : DecidableEq α
      inst✝² : DecidableEq β
      inst✝¹ : DecidableEq γ
      inst✝ : SimpleGraph.TripartiteFromTriangles.NoAccidental t
      s : Finset (Sum α (Sum β γ))
      h : (SimpleGraph.TripartiteFromTriangles.graph t).IsNClique 3 s
      ⊢ Exists fun x => And (Membership.mem t x) (Eq (SimpleGraph.TripartiteFromTria …
    -/
  · rw [is3Clique_iff] at h
    /-
      case refine_1
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      t : Finset (Prod α (Prod β γ))
      inst✝³ : DecidableEq α
      inst✝² : DecidableEq β
      inst✝¹ : DecidableEq γ
      inst✝ : SimpleGraph.TripartiteFromTriangles.NoAccidental t
      s : Finset (Sum α (Sum β γ))
      h : Exists fun a => Exists fun b => Exists fun c => And ((SimpleGraph.Triparti …
      ⊢ Exists fun x => And (Membership.mem t x) (Eq (SimpleGraph.TripartiteFromTria …
    -/
    obtain ⟨x, y, z, hxy, hxz, hyz, rfl⟩ := h
    /-
      case refine_1.intro.intro.intro.intro.intro.intro
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      t : Finset (Prod α (Prod β γ))
      inst✝³ : DecidableEq α
      inst✝² : DecidableEq β
      inst✝¹ : DecidableEq γ
      inst✝ : SimpleGraph.TripartiteFromTriangles.NoAccidental t
      x y z : Sum α (Sum β γ)
      hxy : (SimpleGraph.TripartiteFromTriangles.graph t).Adj x y
      hxz : (SimpleGraph.TripartiteFromTriangles.graph t).Adj x z
      hyz : (SimpleGraph.TripartiteFromTriangles.graph t).Adj y z
      ⊢ Exists fun x_1 => And (Membership.mem t x_1) (Eq (SimpleGraph.TripartiteFrom …
    -/
    obtain ⟨a, b, c, habc, hab, hac, hbc⟩ := graph_triple hxy hxz hyz
    /-
      case refine_1.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intr …
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      t : Finset (Prod α (Prod β γ))
      inst✝³ : DecidableEq α
      inst✝² : DecidableEq β
      inst✝¹ : DecidableEq γ
      inst✝ : SimpleGraph.TripartiteFromTriangles.NoAccidental t
      x y z : Sum α (Sum β γ)
      hxy : (SimpleGraph.TripartiteFromTriangles.graph t).Adj x y
      hxz : (SimpleGraph.TripartiteFromTriangles.graph t).Adj x z
      hyz : (SimpleGraph.TripartiteFromTriangles.graph t).Adj y z
      a : α
      b : β
      c : γ
      habc : Eq (Insert.insert (Sum3.in₀ a) (Insert.insert (Sum3.in₁ b) (Singleton.s …
      hab : (SimpleGraph.TripartiteFromTriangles.graph t).Adj (Sum3.in₀ a) (Sum3.in₁ …
      hac : (SimpleGraph.TripartiteFromTriangles.graph t).Adj (Sum3.in₀ a) (Sum3.in₂ …
      hbc : (SimpleGraph.TripartiteFromTriangles.graph t).Adj (Sum3.in₁ b) (Sum3.in₂ …
      ⊢ Exists fun x_1 => And (Membership.mem t x_1) (Eq (SimpleGraph.TripartiteFrom …
    -/
    refine ⟨(a, b, c), ?_, habc⟩
    /-
      case refine_1.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intr …
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      t : Finset (Prod α (Prod β γ))
      inst✝³ : DecidableEq α
      inst✝² : DecidableEq β
      inst✝¹ : DecidableEq γ
      inst✝ : SimpleGraph.TripartiteFromTriangles.NoAccidental t
      x y z : Sum α (Sum β γ)
      hxy : (SimpleGraph.TripartiteFromTriangles.graph t).Adj x y
      hxz : (SimpleGraph.TripartiteFromTriangles.graph t).Adj x z
      hyz : (SimpleGraph.TripartiteFromTriangles.graph t).Adj y z
      a : α
      b : β
      c : γ
      habc : Eq (Insert.insert (Sum3.in₀ a) (Insert.insert (Sum3.in₁ b) (Singleton.s …
      hab : (SimpleGraph.TripartiteFromTriangles.graph t).Adj (Sum3.in₀ a) (Sum3.in₁ …
      hac : (SimpleGraph.TripartiteFromTriangles.graph t).Adj (Sum3.in₀ a) (Sum3.in₂ …
      hbc : (SimpleGraph.TripartiteFromTriangles.graph t).Adj (Sum3.in₁ b) (Sum3.in₂ …
      ⊢ Membership.mem t { fst := a, snd := { fst := b, snd := c } }
    -/
    obtain ⟨c', hc'⟩ := in₀₁_iff.1 hab
    /-
      case refine_1.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intr …
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      t : Finset (Prod α (Prod β γ))
      inst✝³ : DecidableEq α
      inst✝² : DecidableEq β
      inst✝¹ : DecidableEq γ
      inst✝ : SimpleGraph.TripartiteFromTriangles.NoAccidental t
      x y z : Sum α (Sum β γ)
      hxy : (SimpleGraph.TripartiteFromTriangles.graph t).Adj x y
      hxz : (SimpleGraph.TripartiteFromTriangles.graph t).Adj x z
      hyz : (SimpleGraph.TripartiteFromTriangles.graph t).Adj y z
      a : α
      b : β
      c : γ
      habc : Eq (Insert.insert (Sum3.in₀ a) (Insert.insert (Sum3.in₁ b) (Singleton.s …
      hab : (SimpleGraph.TripartiteFromTriangles.graph t).Adj (Sum3.in₀ a) (Sum3.in₁ …
      hac : (SimpleGraph.TripartiteFromTriangles.graph t).Adj (Sum3.in₀ a) (Sum3.in₂ …
      hbc : (SimpleGraph.TripartiteFromTriangles.graph t).Adj (Sum3.in₁ b) (Sum3.in₂ …
      c' : γ
      hc' : Membership.mem t { fst := a, snd := { fst := b, snd := c' } }
      ⊢ Membership.mem t { fst := a, snd := { fst := b, snd := c } }
    -/
    obtain ⟨b', hb'⟩ := in₀₂_iff.1 hac
    /-
      case refine_1.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intr …
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      t : Finset (Prod α (Prod β γ))
      inst✝³ : DecidableEq α
      inst✝² : DecidableEq β
      inst✝¹ : DecidableEq γ
      inst✝ : SimpleGraph.TripartiteFromTriangles.NoAccidental t
      x y z : Sum α (Sum β γ)
      hxy : (SimpleGraph.TripartiteFromTriangles.graph t).Adj x y
      hxz : (SimpleGraph.TripartiteFromTriangles.graph t).Adj x z
      hyz : (SimpleGraph.TripartiteFromTriangles.graph t).Adj y z
      a : α
      b : β
      c : γ
      habc : Eq (Insert.insert (Sum3.in₀ a) (Insert.insert (Sum3.in₁ b) (Singleton.s …
      hab : (SimpleGraph.TripartiteFromTriangles.graph t).Adj (Sum3.in₀ a) (Sum3.in₁ …
      hac : (SimpleGraph.TripartiteFromTriangles.graph t).Adj (Sum3.in₀ a) (Sum3.in₂ …
      hbc : (SimpleGraph.TripartiteFromTriangles.graph t).Adj (Sum3.in₁ b) (Sum3.in₂ …
      c' : γ
      hc' : Membership.mem t { fst := a, snd := { fst := b, snd := c' } }
      b' : β
      hb' : Membership.mem t { fst := a, snd := { fst := b', snd := c } }
      ⊢ Membership.mem t { fst := a, snd := { fst := b, snd := c } }
    -/
    obtain ⟨a', ha'⟩ := in₁₂_iff.1 hbc
    /-
      case refine_1.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intr …
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      t : Finset (Prod α (Prod β γ))
      inst✝³ : DecidableEq α
      inst✝² : DecidableEq β
      inst✝¹ : DecidableEq γ
      inst✝ : SimpleGraph.TripartiteFromTriangles.NoAccidental t
      x y z : Sum α (Sum β γ)
      hxy : (SimpleGraph.TripartiteFromTriangles.graph t).Adj x y
      hxz : (SimpleGraph.TripartiteFromTriangles.graph t).Adj x z
      hyz : (SimpleGraph.TripartiteFromTriangles.graph t).Adj y z
      a : α
      b : β
      c : γ
      habc : Eq (Insert.insert (Sum3.in₀ a) (Insert.insert (Sum3.in₁ b) (Singleton.s …
      hab : (SimpleGraph.TripartiteFromTriangles.graph t).Adj (Sum3.in₀ a) (Sum3.in₁ …
      hac : (SimpleGraph.TripartiteFromTriangles.graph t).Adj (Sum3.in₀ a) (Sum3.in₂ …
      hbc : (SimpleGraph.TripartiteFromTriangles.graph t).Adj (Sum3.in₁ b) (Sum3.in₂ …
      c' : γ
      hc' : Membership.mem t { fst := a, snd := { fst := b, snd := c' } }
      b' : β
      hb' : Membership.mem t { fst := a, snd := { fst := b', snd := c } }
      a' : α
      ha' : Membership.mem t { fst := a', snd := { fst := b, snd := c } }
      ⊢ Membership.mem t { fst := a, snd := { fst := b, snd := c } }
    -/
                                                                          /-
                                                                            🎉 no goals
                                                                          -/
                                                                          /-
                                                                            🎉 no goals
                                                                          -/
    obtain rfl | rfl | rfl := NoAccidental.eq_or_eq_or_eq ha' hb' hc' <;> assumption
                                                                          /-
                                                                            🎉 no goals
                                                                          -/
    /-
      case refine_2
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      t : Finset (Prod α (Prod β γ))
      inst✝³ : DecidableEq α
      inst✝² : DecidableEq β
      inst✝¹ : DecidableEq γ
      inst✝ : SimpleGraph.TripartiteFromTriangles.NoAccidental t
      s : Finset (Sum α (Sum β γ))
      ⊢ (Exists fun x => And (Membership.mem t x) (Eq (SimpleGraph.TripartiteFromTri …
    -/
  · rintro ⟨x, hx, rfl⟩
    /-
      case refine_2.intro.intro
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      t : Finset (Prod α (Prod β γ))
      inst✝³ : DecidableEq α
      inst✝² : DecidableEq β
      inst✝¹ : DecidableEq γ
      inst✝ : SimpleGraph.TripartiteFromTriangles.NoAccidental t
      x : Prod α (Prod β γ)
      hx : Membership.mem t x
      ⊢ (SimpleGraph.TripartiteFromTriangles.graph t).IsNClique 3 (SimpleGraph.Tripa …
    -/
    exact toTriangle_is3Clique hx
    /-
      🎉 no goals
    -/


lemma toTriangle_surjOn [NoAccidental t] :
    (t : Set (α × β × γ)).SurjOn toTriangle ((graph t).cliqueSet 3) := fun _ ↦ is3Clique_iff.1


lemma map_toTriangle_disjoint [ExplicitDisjoint t] :
    (t.map toTriangle : Set (Finset (α ⊕ β ⊕ γ))).Pairwise
      fun x y ↦ (x ∩ y : Set (α ⊕ β ⊕ γ)).Subsingleton := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    t : Finset (Prod α (Prod β γ))
    inst✝³ : DecidableEq α
    inst✝² : DecidableEq β
    inst✝¹ : DecidableEq γ
    inst✝ : SimpleGraph.TripartiteFromTriangles.ExplicitDisjoint t
    ⊢ (↑(Finset.map SimpleGraph.TripartiteFromTriangles.toTriangle t)).Pairwise fu …
  -/
  intro
  simp only [Finset.coe_map, Set.mem_image, Finset.mem_coe, Prod.exists, Ne,
    forall_exists_index, and_imp]
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    t : Finset (Prod α (Prod β γ))
    inst✝³ : DecidableEq α
    inst✝² : DecidableEq β
    inst✝¹ : DecidableEq γ
    inst✝ : SimpleGraph.TripartiteFromTriangles.ExplicitDisjoint t
    x✝ : Finset (Sum α (Sum β γ))
    ⊢ ∀ (x : α) (x_1 : β) (x_2 : γ), Membership.mem t { fst := x, snd := { fst :=  …
  -/
  rintro a b c habc rfl e x y z hxyz rfl h'
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    t : Finset (Prod α (Prod β γ))
    inst✝³ : DecidableEq α
    inst✝² : DecidableEq β
    inst✝¹ : DecidableEq γ
    inst✝ : SimpleGraph.TripartiteFromTriangles.ExplicitDisjoint t
    a : α
    b : β
    c : γ
    habc : Membership.mem t { fst := a, snd := { fst := b, snd := c } }
    x : α
    y : β
    z : γ
    hxyz : Membership.mem t { fst := x, snd := { fst := y, snd := z } }
    h' : Not (Eq (SimpleGraph.TripartiteFromTriangles.toTriangle { fst := a, snd : …
    ⊢ (Inter.inter ↑(SimpleGraph.TripartiteFromTriangles.toTriangle { fst := a, sn …
  -/
  have := ne_of_apply_ne _ h'
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    t : Finset (Prod α (Prod β γ))
    inst✝³ : DecidableEq α
    inst✝² : DecidableEq β
    inst✝¹ : DecidableEq γ
    inst✝ : SimpleGraph.TripartiteFromTriangles.ExplicitDisjoint t
    a : α
    b : β
    c : γ
    habc : Membership.mem t { fst := a, snd := { fst := b, snd := c } }
    x : α
    y : β
    z : γ
    hxyz : Membership.mem t { fst := x, snd := { fst := y, snd := z } }
    h' : Not (Eq (SimpleGraph.TripartiteFromTriangles.toTriangle { fst := a, snd : …
    this : Ne { fst := a, snd := { fst := b, snd := c } } { fst := x, snd := { fst …
    ⊢ (Inter.inter ↑(SimpleGraph.TripartiteFromTriangles.toTriangle { fst := a, sn …
  -/
  simp only [Ne, Prod.mk.inj_iff, not_and] at this
  simp only [toTriangle_apply, in₀, in₁, in₂, Set.mem_inter_iff, mem_insert, mem_singleton,
    mem_coe, and_imp, Sum.forall, or_false, forall_eq, false_or, eq_self_iff_true, imp_true_iff,
    true_and, and_true, Set.Subsingleton]
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    t : Finset (Prod α (Prod β γ))
    inst✝³ : DecidableEq α
    inst✝² : DecidableEq β
    inst✝¹ : DecidableEq γ
    inst✝ : SimpleGraph.TripartiteFromTriangles.ExplicitDisjoint t
    a : α
    b : β
    c : γ
    habc : Membership.mem t { fst := a, snd := { fst := b, snd := c } }
    x : α
    y : β
    z : γ
    hxyz : Membership.mem t { fst := x, snd := { fst := y, snd := z } }
    h' : Not (Eq (SimpleGraph.TripartiteFromTriangles.toTriangle { fst := a, snd : …
    this : Eq a x → Eq b y → Not (Eq c z)
    ⊢ And (∀ (a_1 : α), Or (Eq (Sum.inl a_1) (Sum.inl a)) (Or (Eq (Sum.inl a_1) (S …
  -/
  suffices ¬ (a = x ∧ b = y) ∧ ¬ (a = x ∧ c = z) ∧ ¬ (b = y ∧ c = z) by aesop
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    t : Finset (Prod α (Prod β γ))
    inst✝³ : DecidableEq α
    inst✝² : DecidableEq β
    inst✝¹ : DecidableEq γ
    inst✝ : SimpleGraph.TripartiteFromTriangles.ExplicitDisjoint t
    a : α
    b : β
    c : γ
    habc : Membership.mem t { fst := a, snd := { fst := b, snd := c } }
    x : α
    y : β
    z : γ
    hxyz : Membership.mem t { fst := x, snd := { fst := y, snd := z } }
    h' : Not (Eq (SimpleGraph.TripartiteFromTriangles.toTriangle { fst := a, snd : …
    this : Eq a x → Eq b y → Not (Eq c z)
    ⊢ And (Not (And (Eq a x) (Eq b y))) (And (Not (And (Eq a x) (Eq c z))) (Not (A …
  -/
  refine ⟨?_, ?_, ?_⟩
    /-
      case refine_1
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      t : Finset (Prod α (Prod β γ))
      inst✝³ : DecidableEq α
      inst✝² : DecidableEq β
      inst✝¹ : DecidableEq γ
      inst✝ : SimpleGraph.TripartiteFromTriangles.ExplicitDisjoint t
      a : α
      b : β
      c : γ
      habc : Membership.mem t { fst := a, snd := { fst := b, snd := c } }
      x : α
      y : β
      z : γ
      hxyz : Membership.mem t { fst := x, snd := { fst := y, snd := z } }
      h' : Not (Eq (SimpleGraph.TripartiteFromTriangles.toTriangle { fst := a, snd : …
      this : Eq a x → Eq b y → Not (Eq c z)
      ⊢ Not (And (Eq a x) (Eq b y))
    -/
  · rintro ⟨rfl, rfl⟩
    /-
      case refine_1.intro
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      t : Finset (Prod α (Prod β γ))
      inst✝³ : DecidableEq α
      inst✝² : DecidableEq β
      inst✝¹ : DecidableEq γ
      inst✝ : SimpleGraph.TripartiteFromTriangles.ExplicitDisjoint t
      a : α
      b : β
      c : γ
      habc : Membership.mem t { fst := a, snd := { fst := b, snd := c } }
      z : γ
      hxyz : Membership.mem t { fst := a, snd := { fst := b, snd := z } }
      h' : Not (Eq (SimpleGraph.TripartiteFromTriangles.toTriangle { fst := a, snd : …
      this : Eq a a → Eq b b → Not (Eq c z)
      ⊢ False
    -/
    exact this rfl rfl (ExplicitDisjoint.inj₂ habc hxyz)
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      t : Finset (Prod α (Prod β γ))
      inst✝³ : DecidableEq α
      inst✝² : DecidableEq β
      inst✝¹ : DecidableEq γ
      inst✝ : SimpleGraph.TripartiteFromTriangles.ExplicitDisjoint t
      a : α
      b : β
      c : γ
      habc : Membership.mem t { fst := a, snd := { fst := b, snd := c } }
      x : α
      y : β
      z : γ
      hxyz : Membership.mem t { fst := x, snd := { fst := y, snd := z } }
      h' : Not (Eq (SimpleGraph.TripartiteFromTriangles.toTriangle { fst := a, snd : …
      this : Eq a x → Eq b y → Not (Eq c z)
      ⊢ Not (And (Eq a x) (Eq c z))
    -/
  · rintro ⟨rfl, rfl⟩
    /-
      case refine_2.intro
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      t : Finset (Prod α (Prod β γ))
      inst✝³ : DecidableEq α
      inst✝² : DecidableEq β
      inst✝¹ : DecidableEq γ
      inst✝ : SimpleGraph.TripartiteFromTriangles.ExplicitDisjoint t
      a : α
      b : β
      c : γ
      habc : Membership.mem t { fst := a, snd := { fst := b, snd := c } }
      y : β
      hxyz : Membership.mem t { fst := a, snd := { fst := y, snd := c } }
      h' : Not (Eq (SimpleGraph.TripartiteFromTriangles.toTriangle { fst := a, snd : …
      this : Eq a a → Eq b y → Not (Eq c c)
      ⊢ False
    -/
    exact this rfl (ExplicitDisjoint.inj₁ habc hxyz) rfl
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      t : Finset (Prod α (Prod β γ))
      inst✝³ : DecidableEq α
      inst✝² : DecidableEq β
      inst✝¹ : DecidableEq γ
      inst✝ : SimpleGraph.TripartiteFromTriangles.ExplicitDisjoint t
      a : α
      b : β
      c : γ
      habc : Membership.mem t { fst := a, snd := { fst := b, snd := c } }
      x : α
      y : β
      z : γ
      hxyz : Membership.mem t { fst := x, snd := { fst := y, snd := z } }
      h' : Not (Eq (SimpleGraph.TripartiteFromTriangles.toTriangle { fst := a, snd : …
      this : Eq a x → Eq b y → Not (Eq c z)
      ⊢ Not (And (Eq b y) (Eq c z))
    -/
  · rintro ⟨rfl, rfl⟩
    /-
      case refine_3.intro
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      t : Finset (Prod α (Prod β γ))
      inst✝³ : DecidableEq α
      inst✝² : DecidableEq β
      inst✝¹ : DecidableEq γ
      inst✝ : SimpleGraph.TripartiteFromTriangles.ExplicitDisjoint t
      a : α
      b : β
      c : γ
      habc : Membership.mem t { fst := a, snd := { fst := b, snd := c } }
      x : α
      hxyz : Membership.mem t { fst := x, snd := { fst := b, snd := c } }
      h' : Not (Eq (SimpleGraph.TripartiteFromTriangles.toTriangle { fst := a, snd : …
      this : Eq a x → Eq b b → Not (Eq c c)
      ⊢ False
    -/
    exact this (ExplicitDisjoint.inj₀ habc hxyz) rfl rfl
    /-
      🎉 no goals
    -/


lemma cliqueSet_eq_image [NoAccidental t] : (graph t).cliqueSet 3 = toTriangle '' t := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    t : Finset (Prod α (Prod β γ))
    inst✝³ : DecidableEq α
    inst✝² : DecidableEq β
    inst✝¹ : DecidableEq γ
    inst✝ : SimpleGraph.TripartiteFromTriangles.NoAccidental t
    ⊢ Eq ((SimpleGraph.TripartiteFromTriangles.graph t).cliqueSet 3) (Set.image ⇑S …
  -/
  ext; exact is3Clique_iff
       /-
         🎉 no goals
       -/


lemma cliqueFinset_eq_image [NoAccidental t] : (graph t).cliqueFinset 3 = t.image toTriangle :=
                      /-
                        α : Type u_1
                        β : Type u_2
                        γ : Type u_3
                        t : Finset (Prod α (Prod β γ))
                        inst✝⁶ : DecidableEq α
                        inst✝⁵ : DecidableEq β
                        inst✝⁴ : DecidableEq γ
                        inst✝³ : Fintype α
                        inst✝² : Fintype β
                        inst✝¹ : Fintype γ
                        inst✝ : SimpleGraph.TripartiteFromTriangles.NoAccidental t
                        ⊢ Eq ↑((SimpleGraph.TripartiteFromTriangles.graph t).cliqueFinset 3) ↑(Finset. …
                      -/
  coe_injective <| by push_cast; exact cliqueSet_eq_image _
                                 /-
                                   🎉 no goals
                                 -/


lemma cliqueFinset_eq_map [NoAccidental t] : (graph t).cliqueFinset 3 = t.map toTriangle := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    t : Finset (Prod α (Prod β γ))
    inst✝⁶ : DecidableEq α
    inst✝⁵ : DecidableEq β
    inst✝⁴ : DecidableEq γ
    inst✝³ : Fintype α
    inst✝² : Fintype β
    inst✝¹ : Fintype γ
    inst✝ : SimpleGraph.TripartiteFromTriangles.NoAccidental t
    ⊢ Eq ((SimpleGraph.TripartiteFromTriangles.graph t).cliqueFinset 3) (Finset.ma …
  -/
  simp [cliqueFinset_eq_image, map_eq_image]
  /-
    🎉 no goals
  -/


@[simp] lemma card_triangles [NoAccidental t] : #((graph t).cliqueFinset 3) = #t := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    t : Finset (Prod α (Prod β γ))
    inst✝⁶ : DecidableEq α
    inst✝⁵ : DecidableEq β
    inst✝⁴ : DecidableEq γ
    inst✝³ : Fintype α
    inst✝² : Fintype β
    inst✝¹ : Fintype γ
    inst✝ : SimpleGraph.TripartiteFromTriangles.NoAccidental t
    ⊢ Eq ((SimpleGraph.TripartiteFromTriangles.graph t).cliqueFinset 3).card t.card
  -/
  rw [cliqueFinset_eq_map, card_map]
  /-
    🎉 no goals
  -/


lemma farFromTriangleFree [ExplicitDisjoint t] {ε : 𝕜}
    (ht : ε * ((Fintype.card α + Fintype.card β + Fintype.card γ) ^ 2 : ℕ) ≤ #t) :
    (graph t).FarFromTriangleFree ε :=
  farFromTriangleFree_of_disjoint_triangles (t.map toTriangle)
                                                    /-
                                                      α : Type u_1
                                                      β : Type u_2
                                                      γ : Type u_3
                                                      𝕜 : Type u_4
                                                      inst✝⁷ : LinearOrderedField 𝕜
                                                      t : Finset (Prod α (Prod β γ))
                                                      inst✝⁶ : DecidableEq α
                                                      inst✝⁵ : DecidableEq β
                                                      inst✝⁴ : DecidableEq γ
                                                      inst✝³ : Fintype α
                                                      inst✝² : Fintype β
                                                      inst✝¹ : Fintype γ
                                                      inst✝ : SimpleGraph.TripartiteFromTriangles.ExplicitDisjoint t
                                                      ε : 𝕜
                                                      ht : LE.le (HMul.hMul ε ↑(HPow.hPow (HAdd.hAdd (HAdd.hAdd (Fintype.card α) (Fi …
                                                      x : Prod α (Prod β γ)
                                                      hx : Membership.mem t x
                                                      ⊢ Membership.mem (((SimpleGraph.TripartiteFromTriangles.graph t).cliqueFinset  …
                                                    -/
    (map_subset_iff_subset_preimage.2 fun x hx ↦ by simpa using toTriangle_is3Clique hx)
                                                    /-
                                                      🎉 no goals
                                                    -/
                                      /-
                                        α : Type u_1
                                        β : Type u_2
                                        γ : Type u_3
                                        𝕜 : Type u_4
                                        inst✝⁷ : LinearOrderedField 𝕜
                                        t : Finset (Prod α (Prod β γ))
                                        inst✝⁶ : DecidableEq α
                                        inst✝⁵ : DecidableEq β
                                        inst✝⁴ : DecidableEq γ
                                        inst✝³ : Fintype α
                                        inst✝² : Fintype β
                                        inst✝¹ : Fintype γ
                                        inst✝ : SimpleGraph.TripartiteFromTriangles.ExplicitDisjoint t
                                        ε : 𝕜
                                        ht : LE.le (HMul.hMul ε ↑(HPow.hPow (HAdd.hAdd (HAdd.hAdd (Fintype.card α) (Fi …
                                        ⊢ LE.le (HMul.hMul ε ↑(HPow.hPow (Fintype.card (Sum α (Sum β γ))) 2)) ↑(Finset …
                                      -/
    (map_toTriangle_disjoint t) <| by simpa [add_assoc] using ht
                                      /-
                                        🎉 no goals
                                      -/


lemma locallyLinear [ExplicitDisjoint t] [NoAccidental t] : (graph t).LocallyLinear := by
  classical
  refine ⟨?_, fun x y hxy ↦ ?_⟩
  · unfold EdgeDisjointTriangles
    convert map_toTriangle_disjoint t
    rw [cliqueSet_eq_image, coe_map]
  · obtain ⟨z, hz, hxy⟩ := exists_mem_toTriangle hxy
    exact ⟨_, toTriangle_is3Clique hz, hxy⟩


