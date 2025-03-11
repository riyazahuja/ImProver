variable (α) in
/-- The **Ruzsa-Szemerédi number** of a fintype is the maximum number of edges a locally linear
graph on that type can have.

In other words, `ruzsaSzemerediNumber α` is the maximum number of edges a graph on `α` can have such
that each edge belongs to exactly one triangle. -/
noncomputable def ruzsaSzemerediNumber : ℕ := by
  classical
  exact Nat.findGreatest (fun m ↦ ∃ (G : SimpleGraph α) (_ : DecidableRel G.Adj),
    #(G.cliqueFinset 3) = m ∧ G.LocallyLinear) ((card α).choose 3)


open scoped Classical in
lemma ruzsaSzemerediNumber_le : ruzsaSzemerediNumber α ≤ (card α).choose 3 := Nat.findGreatest_le _


lemma ruzsaSzemerediNumber_spec :
    ∃ (G : SimpleGraph α) (_ : DecidableRel G.Adj),
      #(G.cliqueFinset 3) = ruzsaSzemerediNumber α ∧ G.LocallyLinear := by
  classical
  exact @Nat.findGreatest_spec _
    (fun m ↦ ∃ (G : SimpleGraph α) (_ : DecidableRel G.Adj),
      #(G.cliqueFinset 3) = m ∧ G.LocallyLinear) _ _ (Nat.zero_le _)
    ⟨⊥, inferInstance, by simp, locallyLinear_bot⟩


lemma SimpleGraph.LocallyLinear.le_ruzsaSzemerediNumber [DecidableRel G.Adj]
    (hG : G.LocallyLinear) : #(G.cliqueFinset 3) ≤ ruzsaSzemerediNumber α := by
  classical
  exact le_findGreatest card_cliqueFinset_le ⟨G, inferInstance, by congr, hG⟩


lemma ruzsaSzemerediNumber_mono (f : α ↪ β) : ruzsaSzemerediNumber α ≤ ruzsaSzemerediNumber β := by
  classical
  refine findGreatest_mono ?_ (choose_mono _ <| Fintype.card_le_of_embedding f)
  rintro n ⟨G, _, rfl, hG⟩
  refine ⟨G.map f, inferInstance, ?_, hG.map _⟩
  rw [← card_map ⟨map f, Finset.map_injective _⟩, ← cliqueFinset_map G f]
  decide


lemma ruzsaSzemerediNumber_congr (e : α ≃ β) : ruzsaSzemerediNumber α = ruzsaSzemerediNumber β :=
  (ruzsaSzemerediNumber_mono (e : α ↪ β)).antisymm <| ruzsaSzemerediNumber_mono e.symm


/-- The `n`-th **Ruzsa-Szemerédi number** is the maximum number of edges a locally linear graph on
`n` vertices can have.

In other words, `ruzsaSzemerediNumberNat n` is the maximum number of edges a graph on `n` vertices
can have such that each edge belongs to exactly one triangle. -/
noncomputable def ruzsaSzemerediNumberNat (n : ℕ) : ℕ := ruzsaSzemerediNumber (Fin n)


@[simp]
lemma ruzsaSzemerediNumberNat_card : ruzsaSzemerediNumberNat (card α) = ruzsaSzemerediNumber α :=
  ruzsaSzemerediNumber_congr (Fintype.equivFin _).symm


lemma ruzsaSzemerediNumberNat_mono : Monotone ruzsaSzemerediNumberNat := fun _m _n h =>
  ruzsaSzemerediNumber_mono (Fin.castLEEmb h)


lemma ruzsaSzemerediNumberNat_le : ruzsaSzemerediNumberNat n ≤ n.choose 3 :=
                                         /-
                                           n : Nat
                                           ⊢ Eq ((Fintype.card (Fin n)).choose 3) (n.choose 3)
                                         -/
  ruzsaSzemerediNumber_le.trans_eq <| by rw [Fintype.card_fin]
                                         /-
                                           🎉 no goals
                                         -/


@[simp] lemma ruzsaSzemerediNumberNat_zero : ruzsaSzemerediNumberNat 0 = 0 :=
  le_zero_iff.1 ruzsaSzemerediNumberNat_le


@[simp] lemma ruzsaSzemerediNumberNat_one : ruzsaSzemerediNumberNat 1 = 0 :=
  le_zero_iff.1 ruzsaSzemerediNumberNat_le


@[simp] lemma ruzsaSzemerediNumberNat_two : ruzsaSzemerediNumberNat 2 = 0 :=
  le_zero_iff.1 ruzsaSzemerediNumberNat_le


/-- The triangle indices for the Ruzsa-Szemerédi construction. -/
private def triangleIndices (s : Finset α) : Finset (α × α × α) :=
  (univ ×ˢ s).map
    ⟨fun xa ↦ (xa.1, xa.1 + xa.2, xa.1 + 2 * xa.2), by
      /-
        α : Type u_1
        β : Type u_2
        inst✝¹ : Fintype α
        inst✝ : CommRing α
        s✝ : Finset α
        x : Prod α (Prod α α)
        s : Finset α
        ⊢ Function.Injective fun xa => { fst := xa.1, snd := { fst := HAdd.hAdd xa.1 x …
      -/
      rintro ⟨x, a⟩ ⟨y, b⟩ h
      /-
        case mk.mk
        α : Type u_1
        β : Type u_2
        inst✝¹ : Fintype α
        inst✝ : CommRing α
        s✝ : Finset α
        x✝ : Prod α (Prod α α)
        s : Finset α
        x a y b : α
        h : Eq ((fun xa => { fst := xa.1, snd := { fst := HAdd.hAdd xa.1 xa.2, snd :=  …
        ⊢ Eq { fst := x, snd := a } { fst := y, snd := b }
      -/
      simp only [Prod.ext_iff] at h
      /-
        case mk.mk
        α : Type u_1
        β : Type u_2
        inst✝¹ : Fintype α
        inst✝ : CommRing α
        s✝ : Finset α
        x✝ : Prod α (Prod α α)
        s : Finset α
        x a y b : α
        h : And (Eq x y) (And (Eq (HAdd.hAdd x a) (HAdd.hAdd y b)) (Eq (HAdd.hAdd x (H …
        ⊢ Eq { fst := x, snd := a } { fst := y, snd := b }
      -/
      obtain rfl := h.1
      /-
        case mk.mk
        α : Type u_1
        β : Type u_2
        inst✝¹ : Fintype α
        inst✝ : CommRing α
        s✝ : Finset α
        x✝ : Prod α (Prod α α)
        s : Finset α
        x a b : α
        h : And (Eq x x) (And (Eq (HAdd.hAdd x a) (HAdd.hAdd x b)) (Eq (HAdd.hAdd x (H …
        ⊢ Eq { fst := x, snd := a } { fst := x, snd := b }
      -/
      obtain rfl := add_right_injective _ h.2.1
      /-
        case mk.mk
        α : Type u_1
        β : Type u_2
        inst✝¹ : Fintype α
        inst✝ : CommRing α
        s✝ : Finset α
        x✝ : Prod α (Prod α α)
        s : Finset α
        x a : α
        h : And (Eq x x) (And (Eq (HAdd.hAdd x a) (HAdd.hAdd x a)) (Eq (HAdd.hAdd x (H …
        ⊢ Eq { fst := x, snd := a } { fst := x, snd := a }
      -/
      rfl⟩
      /-
        🎉 no goals
      -/


@[simp]
private lemma mem_triangleIndices :
                                                                          /-
                                                                            α : Type u_1
                                                                            inst✝¹ : Fintype α
                                                                            inst✝ : CommRing α
                                                                            s : Finset α
                                                                            x : Prod α (Prod α α)
                                                                            ⊢ Iff (Membership.mem (triangleIndices s) x) (Exists fun y => Exists fun a =>  …
                                                                          -/
    x ∈ triangleIndices s ↔ ∃ y, ∃ a ∈ s, (y, y + a, y + 2 * a) = x := by simp [triangleIndices]
                                                                          /-
                                                                            🎉 no goals
                                                                          -/


@[simp]
private lemma card_triangleIndices : #(triangleIndices s) = card α * #s := by
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : CommRing α
    s : Finset α
    ⊢ Eq (triangleIndices s).card (HMul.hMul (Fintype.card α) s.card)
  -/
  simp [triangleIndices, card_univ]
  /-
    🎉 no goals
  -/


private lemma noAccidental (hs : ThreeAPFree (s : Set α)) :
    NoAccidental (triangleIndices s : Finset (α × α × α)) where
  eq_or_eq_or_eq := by
    /-
      α : Type u_1
      inst✝¹ : Fintype α
      inst✝ : CommRing α
      s : Finset α
      hs : ThreeAPFree ↑s
      ⊢ ∀ ⦃a a' b b' c c' : α⦄, Membership.mem (triangleIndices s) { fst := a', snd  …
    -/
    simp only [mem_triangleIndices, Prod.mk.inj_iff, exists_prop, forall_exists_index, and_imp]
    /-
      α : Type u_1
      inst✝¹ : Fintype α
      inst✝ : CommRing α
      s : Finset α
      hs : ThreeAPFree ↑s
      ⊢ ∀ ⦃a a' b b' c c' : α⦄ (x x_1 : α), Membership.mem s x_1 → Eq x a' → Eq (HAd …
    -/
    rintro _ _ _ _ _ _ d a ha rfl rfl rfl b' b hb rfl rfl h₁ d' c hc rfl h₂ rfl
    /-
      α : Type u_1
      inst✝¹ : Fintype α
      inst✝ : CommRing α
      s : Finset α
      hs : ThreeAPFree ↑s
      d a : α
      ha : Membership.mem s a
      b : α
      hb : Membership.mem s b
      d' c : α
      hc : Membership.mem s c
      h₁ : Eq (HAdd.hAdd d' (HMul.hMul 2 b)) (HAdd.hAdd d (HMul.hMul 2 a))
      h₂ : Eq (HAdd.hAdd d' c) (HAdd.hAdd d a)
      ⊢ Or (Eq d' d) (Or (Eq (HAdd.hAdd d a) (HAdd.hAdd d' b)) (Eq (HAdd.hAdd d (HMu …
    -/
    have : a + c = b + b := by linear_combination h₁.symm - h₂.symm
    /-
      α : Type u_1
      inst✝¹ : Fintype α
      inst✝ : CommRing α
      s : Finset α
      hs : ThreeAPFree ↑s
      d a : α
      ha : Membership.mem s a
      b : α
      hb : Membership.mem s b
      d' c : α
      hc : Membership.mem s c
      h₁ : Eq (HAdd.hAdd d' (HMul.hMul 2 b)) (HAdd.hAdd d (HMul.hMul 2 a))
      h₂ : Eq (HAdd.hAdd d' c) (HAdd.hAdd d a)
      this : Eq (HAdd.hAdd a c) (HAdd.hAdd b b)
      ⊢ Or (Eq d' d) (Or (Eq (HAdd.hAdd d a) (HAdd.hAdd d' b)) (Eq (HAdd.hAdd d (HMu …
    -/
    obtain rfl := hs ha hb hc this
    /-
      α : Type u_1
      inst✝¹ : Fintype α
      inst✝ : CommRing α
      s : Finset α
      hs : ThreeAPFree ↑s
      d a : α
      ha : Membership.mem s a
      d' c : α
      hc : Membership.mem s c
      h₂ : Eq (HAdd.hAdd d' c) (HAdd.hAdd d a)
      hb : Membership.mem s a
      h₁ : Eq (HAdd.hAdd d' (HMul.hMul 2 a)) (HAdd.hAdd d (HMul.hMul 2 a))
      this : Eq (HAdd.hAdd a c) (HAdd.hAdd a a)
      ⊢ Or (Eq d' d) (Or (Eq (HAdd.hAdd d a) (HAdd.hAdd d' a)) (Eq (HAdd.hAdd d (HMu …
    -/
    simp_all
    /-
      🎉 no goals
    -/


private instance : ExplicitDisjoint (triangleIndices s : Finset (α × α × α)) where
  inj₀ := by
    /-
      α : Type u_1
      β : Type u_2
      inst✝² : Fintype α
      inst✝¹ : CommRing α
      s : Finset α
      x : Prod α (Prod α α)
      inst✝ : Fact (IsUnit 2)
      ⊢ ∀ ⦃a b c a' : α⦄, Membership.mem (triangleIndices s) { fst := a, snd := { fs …
    -/
    simp only [mem_triangleIndices, Prod.mk.inj_iff, exists_prop, forall_exists_index, and_imp]
    /-
      α : Type u_1
      β : Type u_2
      inst✝² : Fintype α
      inst✝¹ : CommRing α
      s : Finset α
      x : Prod α (Prod α α)
      inst✝ : Fact (IsUnit 2)
      ⊢ ∀ ⦃a b c a' : α⦄ (x x_1 : α), Membership.mem s x_1 → Eq x a → Eq (HAdd.hAdd  …
    -/
    rintro _ _ _ _ x a ha rfl rfl rfl y b hb rfl h₁ h₂
    /-
      α : Type u_1
      β : Type u_2
      inst✝² : Fintype α
      inst✝¹ : CommRing α
      s : Finset α
      x✝ : Prod α (Prod α α)
      inst✝ : Fact (IsUnit 2)
      x a : α
      ha : Membership.mem s a
      y b : α
      hb : Membership.mem s b
      h₁ : Eq (HAdd.hAdd y b) (HAdd.hAdd x a)
      h₂ : Eq (HAdd.hAdd y (HMul.hMul 2 b)) (HAdd.hAdd x (HMul.hMul 2 a))
      ⊢ Eq x y
    -/
    linear_combination 2 * h₁.symm - h₂.symm
    /-
      🎉 no goals
    -/
  inj₁ := by
    /-
      α : Type u_1
      β : Type u_2
      inst✝² : Fintype α
      inst✝¹ : CommRing α
      s : Finset α
      x : Prod α (Prod α α)
      inst✝ : Fact (IsUnit 2)
      ⊢ ∀ ⦃a b c b' : α⦄, Membership.mem (triangleIndices s) { fst := a, snd := { fs …
    -/
    simp only [mem_triangleIndices, Prod.mk.inj_iff, exists_prop, forall_exists_index, and_imp]
    /-
      α : Type u_1
      β : Type u_2
      inst✝² : Fintype α
      inst✝¹ : CommRing α
      s : Finset α
      x : Prod α (Prod α α)
      inst✝ : Fact (IsUnit 2)
      ⊢ ∀ ⦃a b c b' : α⦄ (x x_1 : α), Membership.mem s x_1 → Eq x a → Eq (HAdd.hAdd  …
    -/
    rintro _ _ _ _ x a ha rfl rfl rfl y b hb rfl rfl h
    /-
      α : Type u_1
      β : Type u_2
      inst✝² : Fintype α
      inst✝¹ : CommRing α
      s : Finset α
      x : Prod α (Prod α α)
      inst✝ : Fact (IsUnit 2)
      a : α
      ha : Membership.mem s a
      y b : α
      hb : Membership.mem s b
      h : Eq (HAdd.hAdd y (HMul.hMul 2 b)) (HAdd.hAdd y (HMul.hMul 2 a))
      ⊢ Eq (HAdd.hAdd y a) (HAdd.hAdd y b)
    -/
    simpa [(Fact.out (p := IsUnit (2 : α))).mul_right_inj, eq_comm] using h
    /-
      🎉 no goals
    -/
  inj₂ := by
    /-
      α : Type u_1
      β : Type u_2
      inst✝² : Fintype α
      inst✝¹ : CommRing α
      s : Finset α
      x : Prod α (Prod α α)
      inst✝ : Fact (IsUnit 2)
      ⊢ ∀ ⦃a b c c' : α⦄, Membership.mem (triangleIndices s) { fst := a, snd := { fs …
    -/
    simp only [mem_triangleIndices, Prod.mk.inj_iff, exists_prop, forall_exists_index, and_imp]
    /-
      α : Type u_1
      β : Type u_2
      inst✝² : Fintype α
      inst✝¹ : CommRing α
      s : Finset α
      x : Prod α (Prod α α)
      inst✝ : Fact (IsUnit 2)
      ⊢ ∀ ⦃a b c c' : α⦄ (x x_1 : α), Membership.mem s x_1 → Eq x a → Eq (HAdd.hAdd  …
    -/
    rintro _ _ _ _ x a ha rfl rfl rfl y b hb rfl h rfl
    /-
      α : Type u_1
      β : Type u_2
      inst✝² : Fintype α
      inst✝¹ : CommRing α
      s : Finset α
      x : Prod α (Prod α α)
      inst✝ : Fact (IsUnit 2)
      a : α
      ha : Membership.mem s a
      y b : α
      hb : Membership.mem s b
      h : Eq (HAdd.hAdd y b) (HAdd.hAdd y a)
      ⊢ Eq (HAdd.hAdd y (HMul.hMul 2 a)) (HAdd.hAdd y (HMul.hMul 2 b))
    -/
    simpa [(Fact.out (p := IsUnit (2 : α))).mul_right_inj, eq_comm] using h
    /-
      🎉 no goals
    -/


private lemma locallyLinear (hs : ThreeAPFree (s : Set α)) :
    (graph <| triangleIndices s).LocallyLinear :=
  haveI := noAccidental hs; TripartiteFromTriangles.locallyLinear _


private lemma card_edgeFinset (hs : ThreeAPFree (s : Set α)) [DecidableEq α] :
    #(graph <| triangleIndices s).edgeFinset = 3 * card α * #s := by
  /-
    α : Type u_1
    inst✝³ : Fintype α
    inst✝² : CommRing α
    s : Finset α
    inst✝¹ : Fact (IsUnit 2)
    hs : ThreeAPFree ↑s
    inst✝ : DecidableEq α
    ⊢ Eq (SimpleGraph.TripartiteFromTriangles.graph (triangleIndices s)).edgeFinse …
  -/
  haveI := noAccidental hs
  /-
    α : Type u_1
    inst✝³ : Fintype α
    inst✝² : CommRing α
    s : Finset α
    inst✝¹ : Fact (IsUnit 2)
    hs : ThreeAPFree ↑s
    inst✝ : DecidableEq α
    this : SimpleGraph.TripartiteFromTriangles.NoAccidental (triangleIndices s)
    ⊢ Eq (SimpleGraph.TripartiteFromTriangles.graph (triangleIndices s)).edgeFinse …
  -/
  rw [(locallyLinear hs).card_edgeFinset, card_triangles, card_triangleIndices, mul_assoc]
  /-
    🎉 no goals
  -/


lemma addRothNumber_le_ruzsaSzemerediNumber :
    card α * addRothNumber (univ : Finset α) ≤ ruzsaSzemerediNumber (Sum α (Sum α α)) := by
  /-
    α : Type u_1
    inst✝³ : Fintype α
    inst✝² : DecidableEq α
    inst✝¹ : CommRing α
    inst✝ : Fact (IsUnit 2)
    ⊢ LE.le (HMul.hMul (Fintype.card α) (addRothNumber Finset.univ)) (ruzsaSzemere …
  -/
  obtain ⟨s, -, hscard, hs⟩ := addRothNumber_spec (univ : Finset α)
  /-
    case intro.intro.intro
    α : Type u_1
    inst✝³ : Fintype α
    inst✝² : DecidableEq α
    inst✝¹ : CommRing α
    inst✝ : Fact (IsUnit 2)
    s : Finset α
    hscard : Eq s.card (addRothNumber Finset.univ)
    hs : ThreeAPFree ↑s
    ⊢ LE.le (HMul.hMul (Fintype.card α) (addRothNumber Finset.univ)) (ruzsaSzemere …
  -/
  haveI := noAccidental hs
  /-
    case intro.intro.intro
    α : Type u_1
    inst✝³ : Fintype α
    inst✝² : DecidableEq α
    inst✝¹ : CommRing α
    inst✝ : Fact (IsUnit 2)
    s : Finset α
    hscard : Eq s.card (addRothNumber Finset.univ)
    hs : ThreeAPFree ↑s
    this : SimpleGraph.TripartiteFromTriangles.NoAccidental (triangleIndices s)
    ⊢ LE.le (HMul.hMul (Fintype.card α) (addRothNumber Finset.univ)) (ruzsaSzemere …
  -/
  rw [← hscard, ← card_triangleIndices, ← card_triangles]
  /-
    case intro.intro.intro
    α : Type u_1
    inst✝³ : Fintype α
    inst✝² : DecidableEq α
    inst✝¹ : CommRing α
    inst✝ : Fact (IsUnit 2)
    s : Finset α
    hscard : Eq s.card (addRothNumber Finset.univ)
    hs : ThreeAPFree ↑s
    this : SimpleGraph.TripartiteFromTriangles.NoAccidental (triangleIndices s)
    ⊢ LE.le ((SimpleGraph.TripartiteFromTriangles.graph (triangleIndices s)).cliqu …
  -/
  exact (locallyLinear hs).le_ruzsaSzemerediNumber
  /-
    🎉 no goals
  -/


lemma rothNumberNat_le_ruzsaSzemerediNumberNat (n : ℕ) :
    (2 * n + 1) * rothNumberNat n ≤ ruzsaSzemerediNumberNat (6 * n + 3) := by
  /-
    n : Nat
    ⊢ LE.le (HMul.hMul (HAdd.hAdd (HMul.hMul 2 n) 1) (rothNumberNat n)) (ruzsaSzem …
  -/
  let α := Fin (2 * n + 1)
  /-
    n : Nat
    α : Type := Fin (HAdd.hAdd (HMul.hMul 2 n) 1)
    ⊢ LE.le (HMul.hMul (HAdd.hAdd (HMul.hMul 2 n) 1) (rothNumberNat n)) (ruzsaSzem …
  -/
  have : Nat.Coprime 2 (2 * n + 1) := by simp
  /-
    n : Nat
    α : Type := Fin (HAdd.hAdd (HMul.hMul 2 n) 1)
    this : Nat.Coprime 2 (HAdd.hAdd (HMul.hMul 2 n) 1)
    ⊢ LE.le (HMul.hMul (HAdd.hAdd (HMul.hMul 2 n) 1) (rothNumberNat n)) (ruzsaSzem …
  -/
  haveI : Fact (IsUnit (2 : Fin (2 * n + 1))) := ⟨by simpa using (ZMod.unitOfCoprime 2 this).isUnit⟩
  calc
    (2 * n + 1) * rothNumberNat n
    _ = Fintype.card α * addRothNumber (Iio (n : α)) := by
      rw [Fin.addRothNumber_eq_rothNumberNat le_rfl, Fintype.card_fin]
    _ ≤ Fintype.card α * addRothNumber (univ : Finset α) := by
      gcongr; exact subset_univ _
    _ ≤ ruzsaSzemerediNumber (Sum α (Sum α α)) := addRothNumber_le_ruzsaSzemerediNumber _
    _ = ruzsaSzemerediNumberNat (6 * n + 3) := by
      simp_rw [← ruzsaSzemerediNumberNat_card, Fintype.card_sum, α, Fintype.card_fin]
      ring_nf


/-- Lower bound on the **Ruzsa-Szemerédi problem** in terms of 3AP-free sets.

If there exists a 3AP-free subset of `[1, ..., (n - 3) / 6]` of size `m`, then there exists a graph
with `n` vertices and `(n / 3 - 2) * m` edges such that each edge belongs to exactly one triangle.
-/
theorem rothNumberNat_le_ruzsaSzemerediNumberNat' :
    ∀ n : ℕ, (n / 3 - 2 : ℝ) * rothNumberNat ((n - 3) / 6) ≤ ruzsaSzemerediNumberNat n
            /-
              ⊢ LE.le (HMul.hMul (HSub.hSub (HDiv.hDiv (↑0) 3) 2) ↑(rothNumberNat (HDiv.hDiv …
            -/
  | 0 => by simp
            /-
              🎉 no goals
            -/
            /-
              ⊢ LE.le (HMul.hMul (HSub.hSub (HDiv.hDiv (↑1) 3) 2) ↑(rothNumberNat (HDiv.hDiv …
            -/
  | 1 => by simp
            /-
              🎉 no goals
            -/
            /-
              ⊢ LE.le (HMul.hMul (HSub.hSub (HDiv.hDiv (↑2) 3) 2) ↑(rothNumberNat (HDiv.hDiv …
            -/
  | 2 => by simp
            /-
              🎉 no goals
            -/
  | n + 3 => by
    calc
      _ ≤ (↑(2 * (n / 6) + 1) : ℝ) * rothNumberNat (n / 6) :=
        mul_le_mul_of_nonneg_right ?_ (Nat.cast_nonneg _)
      _ ≤ (ruzsaSzemerediNumberNat (6 * (n / 6) + 3) : ℝ) := ?_
      _ ≤ _ :=
        Nat.cast_le.2 (ruzsaSzemerediNumberNat_mono <| add_le_add_right (Nat.mul_div_le _ _) _)
      /-
        case calc_1
        n : Nat
        ⊢ LE.le (HSub.hSub (HDiv.hDiv (↑(HAdd.hAdd n 3)) 3) 2) ↑(HAdd.hAdd (HMul.hMul  …
      -/
    · norm_num
      rw [← div_add_one (three_ne_zero' ℝ), ← le_sub_iff_add_le, div_le_iff₀ (zero_lt_three' ℝ),
        add_assoc, add_sub_assoc, add_mul, mul_right_comm]
      /-
        case calc_1
        n : Nat
        ⊢ LE.le (↑n) (HAdd.hAdd (HMul.hMul (HMul.hMul 2 3) ↑(HDiv.hDiv n 6)) (HMul.hMu …
      -/
      norm_num
      /-
        case calc_1
        n : Nat
        ⊢ LE.le (↑n) (HAdd.hAdd (HMul.hMul 6 ↑(HDiv.hDiv n 6)) 6)
      -/
      norm_cast
      /-
        case calc_1
        n : Nat
        ⊢ LE.le n (HAdd.hAdd (HMul.hMul 6 (HDiv.hDiv n 6)) 6)
      -/
      rw [← mul_add_one]
      /-
        case calc_1
        n : Nat
        ⊢ LE.le n (HMul.hMul 6 (HAdd.hAdd (HDiv.hDiv n 6) 1))
      -/
      exact (Nat.lt_mul_div_succ _ <| by norm_num).le
      /-
        🎉 no goals
      -/
      /-
        case calc_2
        n : Nat
        ⊢ LE.le (HMul.hMul ↑(HAdd.hAdd (HMul.hMul 2 (HDiv.hDiv n 6)) 1) ↑(rothNumberNa …
      -/
    · norm_cast
      /-
        case calc_2
        n : Nat
        ⊢ LE.le (HMul.hMul (HAdd.hAdd (HMul.hMul 2 (HDiv.hDiv n 6)) 1) (rothNumberNat  …
      -/
      exact rothNumberNat_le_ruzsaSzemerediNumberNat _
      /-
        🎉 no goals
      -/


/-- Explicit lower bound on the **Ruzsa-Szemerédi problem**.

There exists a graph with `n` vertices and
`(n / 3 - 2) * (n - 3) / 6 * exp (-4 * sqrt (log ((n - 3) / 6)))` edges such that each edge belongs
to exactly one triangle. -/
theorem ruzsaSzemerediNumberNat_lower_bound (n : ℕ) :
    (n / 3 - 2 : ℝ) * ↑((n - 3) / 6) * exp (-4 * sqrt (log ↑((n - 3) / 6))) ≤
      ruzsaSzemerediNumberNat n := by
  /-
    n : Nat
    ⊢ LE.le (HMul.hMul (HMul.hMul (HSub.hSub (HDiv.hDiv (↑n) 3) 2) ↑(HDiv.hDiv (HS …
  -/
  rw [mul_assoc]
  /-
    n : Nat
    ⊢ LE.le (HMul.hMul (HSub.hSub (HDiv.hDiv (↑n) 3) 2) (HMul.hMul (↑(HDiv.hDiv (H …
  -/
  obtain hn | hn := le_total (n / 3 - 2 : ℝ) 0
    /-
      case inl
      n : Nat
      hn : LE.le (HSub.hSub (HDiv.hDiv (↑n) 3) 2) 0
      ⊢ LE.le (HMul.hMul (HSub.hSub (HDiv.hDiv (↑n) 3) 2) (HMul.hMul (↑(HDiv.hDiv (H …
    -/
  · exact (mul_nonpos_of_nonpos_of_nonneg hn <| by positivity).trans (Nat.cast_nonneg _)
    /-
      🎉 no goals
    -/
  exact
    (mul_le_mul_of_nonneg_left Behrend.roth_lower_bound hn).trans
      (rothNumberNat_le_ruzsaSzemerediNumberNat' _)


/-- Asymptotic lower bound on the **Ruzsa-Szemerédi problem**.

There exists a graph with `n` vertices and `Ω((n ^ 2 * exp (-4 * sqrt (log n))))` edges such that
each edge belongs to exactly one triangle. -/
theorem ruzsaSzemerediNumberNat_asymptotic_lower_bound :
   (fun n ↦ n ^ 2 * exp (-4 * sqrt (log n)) : ℕ → ℝ) =O[atTop]
     fun n ↦ (ruzsaSzemerediNumberNat n : ℝ) := by
  /-
    ⊢ Asymptotics.IsBigO Filter.atTop (fun n => HMul.hMul (HPow.hPow (↑n) 2) (Real …
  -/
  trans fun n ↦ (n / 3 - 2) * ↑((n - 3) / 6) * exp (-4 * sqrt (log ↑((n - 3) / 6)))
    /-
      ⊢ Asymptotics.IsBigO Filter.atTop (fun n => HMul.hMul (HPow.hPow (↑n) 2) (Real …
    -/
  · simp_rw [sq]
    /-
      ⊢ Asymptotics.IsBigO Filter.atTop (fun n => HMul.hMul (HMul.hMul ↑n ↑n) (Real. …
    -/
    refine (IsBigO.mul ?_ ?_).mul ?_
      /-
        case refine_1
        ⊢ Asymptotics.IsBigO Filter.atTop Nat.cast fun n => HSub.hSub (HDiv.hDiv (↑n)  …
      -/
    · trans fun n ↦ n / 3
        /-
          ⊢ Asymptotics.IsBigO Filter.atTop Nat.cast fun n => HDiv.hDiv (↑n) 3
        -/
      · simp_rw [div_eq_inv_mul]
        /-
          ⊢ Asymptotics.IsBigO Filter.atTop Nat.cast fun n => HMul.hMul (Inv.inv 3) ↑n
        -/
        exact (isBigO_refl ..).const_mul_right (by norm_num)
        /-
          🎉 no goals
        -/
      /-
        ⊢ Asymptotics.IsBigO Filter.atTop (fun n => HDiv.hDiv (↑n) 3) fun n => HSub.hS …
      -/
      refine IsLittleO.right_isBigO_sub ?_
      simpa [div_eq_inv_mul, Function.comp_def] using
        .atTop_of_const_mul zero_lt_three (by simp [tendsto_natCast_atTop_atTop])
      /-
        case refine_2
        ⊢ Asymptotics.IsBigO Filter.atTop Nat.cast fun n => ↑(HDiv.hDiv (HSub.hSub n 3 …
      -/
    · rw [IsBigO_def]
      /-
        case refine_2
        ⊢ Exists fun c => Asymptotics.IsBigOWith c Filter.atTop Nat.cast fun n => ↑(HD …
      -/
      refine ⟨12, ?_⟩
      /-
        case refine_2
        ⊢ Asymptotics.IsBigOWith 12 Filter.atTop Nat.cast fun n => ↑(HDiv.hDiv (HSub.h …
      -/
      simp only [IsBigOWith, norm_natCast, eventually_atTop]
      /-
        case refine_2
        ⊢ Exists fun a => ∀ (b : Nat), GE.ge b a → LE.le (↑b) (HMul.hMul 12 ↑(HDiv.hDi …
      -/
      exact ⟨15, fun x hx ↦ by norm_cast; omega⟩
      /-
        🎉 no goals
      -/
      /-
        case refine_3
        ⊢ Asymptotics.IsBigO Filter.atTop (fun n => Real.exp (HMul.hMul (-4) (Real.log …
      -/
    · rw [isBigO_exp_comp_exp_comp]
      /-
        case refine_3
        ⊢ Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) Filter.atTop (HSub.hSub (fu …
      -/
      refine ⟨0, ?_⟩
      simp only [neg_mul, eventually_map, Pi.sub_apply, sub_neg_eq_add, neg_add_le_iff_le_add,
        add_zero, ofNat_pos, _root_.mul_le_mul_left, eventually_atTop]
      /-
        case refine_3
        ⊢ Exists fun a => ∀ (b : Nat), GE.ge b a → LE.le (Real.log ↑(HDiv.hDiv (HSub.h …
      -/
      refine ⟨9, fun x hx ↦ ?_⟩
      /-
        case refine_3
        x : Nat
        hx : GE.ge x 9
        ⊢ LE.le (Real.log ↑(HDiv.hDiv (HSub.hSub x 3) 6)).sqrt (Real.log ↑x).sqrt
      -/
      gcongr
        /-
          case refine_3.h.hx
          x : Nat
          hx : GE.ge x 9
          ⊢ LT.lt 0 ↑(HDiv.hDiv (HSub.hSub x 3) 6)
        -/
      · simp
        /-
          case refine_3.h.hx
          x : Nat
          hx : GE.ge x 9
          ⊢ LE.le 6 (HSub.hSub x 3)
        -/
        omega
        /-
          🎉 no goals
        -/
        /-
          case refine_3.h.hxy.h
          x : Nat
          hx : GE.ge x 9
          ⊢ LE.le (HDiv.hDiv (HSub.hSub x 3) 6) x
        -/
      · omega
        /-
          🎉 no goals
        -/
    /-
      ⊢ Asymptotics.IsBigO Filter.atTop (fun n => HMul.hMul (HMul.hMul (HSub.hSub (H …
    -/
  · refine .of_bound 1 ?_
    /-
      ⊢ Filter.Eventually (fun x => LE.le (Norm.norm (HMul.hMul (HMul.hMul (HSub.hSu …
    -/
    simp only [neg_mul, norm_eq_abs, norm_natCast, one_mul, eventually_atTop]
    /-
      ⊢ Exists fun a => ∀ (b : Nat), GE.ge b a → LE.le (abs (HMul.hMul (HMul.hMul (H …
    -/
    refine ⟨6, fun n hn ↦ ?_⟩
    /-
      n : Nat
      hn : GE.ge n 6
      ⊢ LE.le (abs (HMul.hMul (HMul.hMul (HSub.hSub (HDiv.hDiv (↑n) 3) 2) ↑(HDiv.hDi …
    -/
    have : (0 : ℝ) ≤ n / 3 - 2 := by rify at hn; linarith
    /-
      n : Nat
      hn : GE.ge n 6
      this : LE.le 0 (HSub.hSub (HDiv.hDiv (↑n) 3) 2)
      ⊢ LE.le (abs (HMul.hMul (HMul.hMul (HSub.hSub (HDiv.hDiv (↑n) 3) 2) ↑(HDiv.hDi …
    -/
    simpa using abs_le_abs_of_nonneg (by positivity) (ruzsaSzemerediNumberNat_lower_bound n)
    /-
      🎉 no goals
    -/

