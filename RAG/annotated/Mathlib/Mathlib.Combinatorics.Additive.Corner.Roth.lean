/-- The triangle indices for the proof of the corners theorem construction. -/
private def triangleIndices (A : Finset (G × G)) : Finset (G × G × G) :=
                                        /-
                                          G : Type u_1
                                          inst✝ : AddCommGroup G
                                          A✝ : Finset (Prod G G)
                                          a b c : G
                                          n : Nat
                                          ε : Real
                                          A : Finset (Prod G G)
                                          ⊢ Function.Injective fun x => Corners.triangleIndices.match_1 (fun x => Prod G …
                                        -/
  A.map ⟨fun (a, b) ↦ (a, b, a + b), by rintro ⟨x₁, x₂⟩ ⟨y₁, y₂⟩ ⟨⟩; rfl⟩
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


@[simp]
private lemma mk_mem_triangleIndices : (a, b, c) ∈ triangleIndices A ↔ (a, b) ∈ A ∧ c = a + b := by
  simp only [triangleIndices, Prod.ext_iff, mem_map, Embedding.coeFn_mk, exists_prop, Prod.exists,
    eq_comm]
  /-
    G : Type u_1
    inst✝ : AddCommGroup G
    A : Finset (Prod G G)
    a b c : G
    ⊢ Iff (Exists fun a_1 => Exists fun b_1 => And (Membership.mem A { fst := a_1, …
  -/
  refine ⟨?_, fun h ↦ ⟨_, _, h.1, rfl, rfl, h.2⟩⟩
  /-
    G : Type u_1
    inst✝ : AddCommGroup G
    A : Finset (Prod G G)
    a b c : G
    ⊢ (Exists fun a_1 => Exists fun b_1 => And (Membership.mem A { fst := a_1, snd …
  -/
  rintro ⟨_, _, h₁, rfl, rfl, h₂⟩
  /-
    case intro.intro.intro.intro.intro
    G : Type u_1
    inst✝ : AddCommGroup G
    A : Finset (Prod G G)
    a b c : G
    h₁ : Membership.mem A { fst := a, snd := b }
    h₂ : Eq c (HAdd.hAdd a b)
    ⊢ And (Membership.mem A { fst := a, snd := b }) (Eq c (HAdd.hAdd a b))
  -/
  exact ⟨h₁, h₂⟩
  /-
    🎉 no goals
  -/


@[simp] private lemma card_triangleIndices : #(triangleIndices A) = #A := card_map _


private instance triangleIndices.instExplicitDisjoint : ExplicitDisjoint (triangleIndices A) := by
  /-
    G : Type u_1
    inst✝ : AddCommGroup G
    A : Finset (Prod G G)
    a b c : G
    n : Nat
    ε : Real
    ⊢ SimpleGraph.TripartiteFromTriangles.ExplicitDisjoint (Corners.triangleIndice …
  -/
  constructor
  all_goals
    simp only [mk_mem_triangleIndices, Prod.mk.inj_iff, exists_prop, forall_exists_index, and_imp]
    rintro a b _ a' - rfl - h'
    simp [Fin.val_eq_val, *] at * <;> assumption


private lemma noAccidental (hs : IsCornerFree (A : Set (G × G))) :
    NoAccidental (triangleIndices A) where
  eq_or_eq_or_eq a a' b b' c c' ha hb hc := by
    /-
      G : Type u_1
      inst✝ : AddCommGroup G
      A : Finset (Prod G G)
      hs : IsCornerFree ↑A
      a a' b b' c c' : G
      ha : Membership.mem (Corners.triangleIndices A) { fst := a', snd := { fst := b …
      hb : Membership.mem (Corners.triangleIndices A) { fst := a, snd := { fst := b' …
      hc : Membership.mem (Corners.triangleIndices A) { fst := a, snd := { fst := b, …
      ⊢ Or (Eq a a') (Or (Eq b b') (Eq c c'))
    -/
    simp only [mk_mem_triangleIndices] at ha hb hc
    /-
      G : Type u_1
      inst✝ : AddCommGroup G
      A : Finset (Prod G G)
      hs : IsCornerFree ↑A
      a a' b b' c c' : G
      ha : And (Membership.mem A { fst := a', snd := b }) (Eq c (HAdd.hAdd a' b))
      hb : And (Membership.mem A { fst := a, snd := b' }) (Eq c (HAdd.hAdd a b'))
      hc : And (Membership.mem A { fst := a, snd := b }) (Eq c' (HAdd.hAdd a b))
      ⊢ Or (Eq a a') (Or (Eq b b') (Eq c c'))
    -/
    exact .inl <| hs ⟨hc.1, hb.1, ha.1, hb.2.symm.trans ha.2⟩
    /-
      🎉 no goals
    -/


private lemma farFromTriangleFree_graph [Fintype G] [DecidableEq G] (hε : ε * card G ^ 2 ≤ #A) :
    (graph <| triangleIndices A).FarFromTriangleFree (ε / 9) := by
  /-
    G : Type u_1
    inst✝² : AddCommGroup G
    A : Finset (Prod G G)
    ε : Real
    inst✝¹ : Fintype G
    inst✝ : DecidableEq G
    hε : LE.le (HMul.hMul ε (HPow.hPow (↑(Fintype.card G)) 2)) ↑A.card
    ⊢ (SimpleGraph.TripartiteFromTriangles.graph (Corners.triangleIndices A)).FarF …
  -/
  refine farFromTriangleFree _ ?_
  /-
    G : Type u_1
    inst✝² : AddCommGroup G
    A : Finset (Prod G G)
    ε : Real
    inst✝¹ : Fintype G
    inst✝ : DecidableEq G
    hε : LE.le (HMul.hMul ε (HPow.hPow (↑(Fintype.card G)) 2)) ↑A.card
    ⊢ LE.le (HMul.hMul (HDiv.hDiv ε 9) ↑(HPow.hPow (HAdd.hAdd (HAdd.hAdd (Fintype. …
  -/
  simp_rw [card_triangleIndices, mul_comm_div, Nat.cast_pow, Nat.cast_add]
  /-
    G : Type u_1
    inst✝² : AddCommGroup G
    A : Finset (Prod G G)
    ε : Real
    inst✝¹ : Fintype G
    inst✝ : DecidableEq G
    hε : LE.le (HMul.hMul ε (HPow.hPow (↑(Fintype.card G)) 2)) ↑A.card
    ⊢ LE.le (HMul.hMul ε (HDiv.hDiv (HPow.hPow (HAdd.hAdd (HAdd.hAdd ↑(Fintype.car …
  -/
  ring_nf
  /-
    G : Type u_1
    inst✝² : AddCommGroup G
    A : Finset (Prod G G)
    ε : Real
    inst✝¹ : Fintype G
    inst✝ : DecidableEq G
    hε : LE.le (HMul.hMul ε (HPow.hPow (↑(Fintype.card G)) 2)) ↑A.card
    ⊢ LE.le (HMul.hMul ε (HPow.hPow (↑(Fintype.card G)) 2)) ↑A.card
  -/
  simpa only [mul_comm] using hε
  /-
    🎉 no goals
  -/


/-- An explicit form for the constant in the corners theorem.

Note that this depends on `SzemerediRegularity.bound`, which is a tower-type exponential. This means
`cornersTheoremBound` is in practice absolutely tiny. -/
noncomputable def cornersTheoremBound (ε : ℝ) : ℕ := ⌊(triangleRemovalBound (ε / 9) * 27)⁻¹⌋₊ + 1


/-- The **corners theorem** for finite abelian groups.

The maximum density of a corner-free set in `G × G` goes to zero as `|G|` tends to infinity. -/
theorem corners_theorem (ε : ℝ) (hε : 0 < ε) (hG : cornersTheoremBound ε ≤ card G)
    (A : Finset (G × G)) (hAε : ε * card G ^ 2 ≤ #A) : ¬ IsCornerFree (A : Set (G × G)) := by
  /-
    G : Type u_1
    inst✝¹ : AddCommGroup G
    inst✝ : Fintype G
    ε : Real
    hε : LT.lt 0 ε
    hG : LE.le (cornersTheoremBound ε) (Fintype.card G)
    A : Finset (Prod G G)
    hAε : LE.le (HMul.hMul ε (HPow.hPow (↑(Fintype.card G)) 2)) ↑A.card
    ⊢ Not (IsCornerFree ↑A)
  -/
  rintro hA
  /-
    G : Type u_1
    inst✝¹ : AddCommGroup G
    inst✝ : Fintype G
    ε : Real
    hε : LT.lt 0 ε
    hG : LE.le (cornersTheoremBound ε) (Fintype.card G)
    A : Finset (Prod G G)
    hAε : LE.le (HMul.hMul ε (HPow.hPow (↑(Fintype.card G)) 2)) ↑A.card
    hA : IsCornerFree ↑A
    ⊢ False
  -/
  rw [cornersTheoremBound, Nat.add_one_le_iff] at hG
  have hε₁ : ε ≤ 1 := by
    have := hAε.trans (Nat.cast_le.2 A.card_le_univ)
    simp only [sq, Nat.cast_mul, Fintype.card_prod, Fintype.card_fin] at this
    rwa [mul_le_iff_le_one_left] at this
    positivity
  /-
    G : Type u_1
    inst✝¹ : AddCommGroup G
    inst✝ : Fintype G
    ε : Real
    hε : LT.lt 0 ε
    hG : LT.lt (Nat.floor (Inv.inv (HMul.hMul (SimpleGraph.triangleRemovalBound (H …
    A : Finset (Prod G G)
    hAε : LE.le (HMul.hMul ε (HPow.hPow (↑(Fintype.card G)) 2)) ↑A.card
    hA : IsCornerFree ↑A
    hε₁ : LE.le ε 1
    ⊢ False
  -/
  have := noAccidental hA
  /-
    G : Type u_1
    inst✝¹ : AddCommGroup G
    inst✝ : Fintype G
    ε : Real
    hε : LT.lt 0 ε
    hG : LT.lt (Nat.floor (Inv.inv (HMul.hMul (SimpleGraph.triangleRemovalBound (H …
    A : Finset (Prod G G)
    hAε : LE.le (HMul.hMul ε (HPow.hPow (↑(Fintype.card G)) 2)) ↑A.card
    hA : IsCornerFree ↑A
    hε₁ : LE.le ε 1
    this : SimpleGraph.TripartiteFromTriangles.NoAccidental (Corners.triangleIndic …
    ⊢ False
  -/
  rw [Nat.floor_lt' (by positivity), inv_lt_iff_one_lt_mul₀'] at hG
  /-
    G : Type u_1
    inst✝¹ : AddCommGroup G
    inst✝ : Fintype G
    ε : Real
    hε : LT.lt 0 ε
    hG : LT.lt 1 (HMul.hMul (HMul.hMul (SimpleGraph.triangleRemovalBound (HDiv.hDi …
    A : Finset (Prod G G)
    hAε : LE.le (HMul.hMul ε (HPow.hPow (↑(Fintype.card G)) 2)) ↑A.card
    hA : IsCornerFree ↑A
    hε₁ : LE.le ε 1
    this : SimpleGraph.TripartiteFromTriangles.NoAccidental (Corners.triangleIndic …
    ⊢ False
  -/
  swap
    /-
      G : Type u_1
      inst✝¹ : AddCommGroup G
      inst✝ : Fintype G
      ε : Real
      hε : LT.lt 0 ε
      hG : LT.lt (Inv.inv (HMul.hMul (SimpleGraph.triangleRemovalBound (HDiv.hDiv ε  …
      A : Finset (Prod G G)
      hAε : LE.le (HMul.hMul ε (HPow.hPow (↑(Fintype.card G)) 2)) ↑A.card
      hA : IsCornerFree ↑A
      hε₁ : LE.le ε 1
      this : SimpleGraph.TripartiteFromTriangles.NoAccidental (Corners.triangleIndic …
      ⊢ LT.lt 0 (HMul.hMul (SimpleGraph.triangleRemovalBound (HDiv.hDiv ε 9)) 27)
    -/
  · have : ε / 9 ≤ 1 := by linarith
    /-
      G : Type u_1
      inst✝¹ : AddCommGroup G
      inst✝ : Fintype G
      ε : Real
      hε : LT.lt 0 ε
      hG : LT.lt (Inv.inv (HMul.hMul (SimpleGraph.triangleRemovalBound (HDiv.hDiv ε  …
      A : Finset (Prod G G)
      hAε : LE.le (HMul.hMul ε (HPow.hPow (↑(Fintype.card G)) 2)) ↑A.card
      hA : IsCornerFree ↑A
      hε₁ : LE.le ε 1
      this✝ : SimpleGraph.TripartiteFromTriangles.NoAccidental (Corners.triangleIndi …
      this : LE.le (HDiv.hDiv ε 9) 1
      ⊢ LT.lt 0 (HMul.hMul (SimpleGraph.triangleRemovalBound (HDiv.hDiv ε 9)) 27)
    -/
    positivity
    /-
      🎉 no goals
    -/
  /-
    G : Type u_1
    inst✝¹ : AddCommGroup G
    inst✝ : Fintype G
    ε : Real
    hε : LT.lt 0 ε
    hG : LT.lt 1 (HMul.hMul (HMul.hMul (SimpleGraph.triangleRemovalBound (HDiv.hDi …
    A : Finset (Prod G G)
    hAε : LE.le (HMul.hMul ε (HPow.hPow (↑(Fintype.card G)) 2)) ↑A.card
    hA : IsCornerFree ↑A
    hε₁ : LE.le ε 1
    this : SimpleGraph.TripartiteFromTriangles.NoAccidental (Corners.triangleIndic …
    ⊢ False
  -/
  refine hG.not_le (le_of_mul_le_mul_right ?_ (by positivity : (0 : ℝ) < card G ^ 2))
  classical
  have h₁ := (farFromTriangleFree_graph hAε).le_card_cliqueFinset
  rw [card_triangles, card_triangleIndices] at h₁
  convert h₁.trans (Nat.cast_le.2 <| card_le_univ _) using 1 <;> simp <;> ring


/-- The **corners theorem** for `ℕ`.

The maximum density of a corner-free set in `{1, ..., n} × {1, ..., n}` goes to zero as `n` tends to
infinity. -/
theorem corners_theorem_nat (hε : 0 < ε) (hn : cornersTheoremBound (ε / 9) ≤ n)
    (A : Finset (ℕ × ℕ)) (hAn : A ⊆ range n ×ˢ range n) (hAε : ε * n ^ 2 ≤ #A) :
    ¬ IsCornerFree (A : Set (ℕ × ℕ)) := by
  /-
    n : Nat
    ε : Real
    hε : LT.lt 0 ε
    hn : LE.le (cornersTheoremBound (HDiv.hDiv ε 9)) n
    A : Finset (Prod Nat Nat)
    hAn : HasSubset.Subset A (SProd.sprod (Finset.range n) (Finset.range n))
    hAε : LE.le (HMul.hMul ε (HPow.hPow (↑n) 2)) ↑A.card
    ⊢ Not (IsCornerFree ↑A)
  -/
  rintro hA
  /-
    n : Nat
    ε : Real
    hε : LT.lt 0 ε
    hn : LE.le (cornersTheoremBound (HDiv.hDiv ε 9)) n
    A : Finset (Prod Nat Nat)
    hAn : HasSubset.Subset A (SProd.sprod (Finset.range n) (Finset.range n))
    hAε : LE.le (HMul.hMul ε (HPow.hPow (↑n) 2)) ↑A.card
    hA : IsCornerFree ↑A
    ⊢ False
  -/
  rw [← coe_subset, coe_product] at hAn
  have : A = Prod.map Fin.val Fin.val ''
      (Prod.map Nat.cast Nat.cast '' A : Set (Fin (2 * n).succ × Fin (2 * n).succ)) := by
    rw [Set.image_image, Set.image_congr, Set.image_id]
    simp only [mem_coe, Nat.succ_eq_add_one, Prod.map_apply, Fin.val_natCast, id_eq, Prod.forall,
      Prod.mk.injEq, Nat.mod_succ_eq_iff_lt]
    rintro a b hab
    have := hAn hab
    simp at this
    omega
  /-
    n : Nat
    ε : Real
    hε : LT.lt 0 ε
    hn : LE.le (cornersTheoremBound (HDiv.hDiv ε 9)) n
    A : Finset (Prod Nat Nat)
    hAn : HasSubset.Subset (↑A) (SProd.sprod ↑(Finset.range n) ↑(Finset.range n))
    hAε : LE.le (HMul.hMul ε (HPow.hPow (↑n) 2)) ↑A.card
    hA : IsCornerFree ↑A
    this : Eq (↑A) (Set.image (Prod.map Fin.val Fin.val) (Set.image (Prod.map Nat. …
    ⊢ False
  -/
  rw [this] at hA
  /-
    n : Nat
    ε : Real
    hε : LT.lt 0 ε
    hn : LE.le (cornersTheoremBound (HDiv.hDiv ε 9)) n
    A : Finset (Prod Nat Nat)
    hAn : HasSubset.Subset (↑A) (SProd.sprod ↑(Finset.range n) ↑(Finset.range n))
    hAε : LE.le (HMul.hMul ε (HPow.hPow (↑n) 2)) ↑A.card
    hA : IsCornerFree (Set.image (Prod.map Fin.val Fin.val) (Set.image (Prod.map N …
    this : Eq (↑A) (Set.image (Prod.map Fin.val Fin.val) (Set.image (Prod.map Nat. …
    ⊢ False
  -/
  have := Fin.isAddFreimanIso_Iio two_ne_zero (le_refl (2 * n))
  have := hA.of_image this.isAddFreimanHom Fin.val_injective.injOn <| by
    refine Set.image_subset_iff.2 <| hAn.trans fun x hx ↦ ?_
    simp only [coe_range, Set.mem_prod, Set.mem_Iio] at hx
    exact ⟨Fin.natCast_strictMono (by omega) hx.1, Fin.natCast_strictMono (by omega) hx.2⟩
  /-
    n : Nat
    ε : Real
    hε : LT.lt 0 ε
    hn : LE.le (cornersTheoremBound (HDiv.hDiv ε 9)) n
    A : Finset (Prod Nat Nat)
    hAn : HasSubset.Subset (↑A) (SProd.sprod ↑(Finset.range n) ↑(Finset.range n))
    hAε : LE.le (HMul.hMul ε (HPow.hPow (↑n) 2)) ↑A.card
    hA : IsCornerFree (Set.image (Prod.map Fin.val Fin.val) (Set.image (Prod.map N …
    this✝¹ : Eq (↑A) (Set.image (Prod.map Fin.val Fin.val) (Set.image (Prod.map Na …
    this✝ : IsAddFreimanIso 2 (Set.Iio ↑n) (Set.Iio n) Fin.val
    this : IsCornerFree (Set.image (Prod.map Nat.cast Nat.cast) ↑A)
    ⊢ False
  -/
  rw [← coe_image] at this
  /-
    n : Nat
    ε : Real
    hε : LT.lt 0 ε
    hn : LE.le (cornersTheoremBound (HDiv.hDiv ε 9)) n
    A : Finset (Prod Nat Nat)
    hAn : HasSubset.Subset (↑A) (SProd.sprod ↑(Finset.range n) ↑(Finset.range n))
    hAε : LE.le (HMul.hMul ε (HPow.hPow (↑n) 2)) ↑A.card
    hA : IsCornerFree (Set.image (Prod.map Fin.val Fin.val) (Set.image (Prod.map N …
    this✝¹ : Eq (↑A) (Set.image (Prod.map Fin.val Fin.val) (Set.image (Prod.map Na …
    this✝ : IsAddFreimanIso 2 (Set.Iio ↑n) (Set.Iio n) Fin.val
    this : IsCornerFree ↑(Finset.image (Prod.map Nat.cast Nat.cast) A)
    ⊢ False
  -/
  refine corners_theorem (ε / 9) (by positivity) (by simp; omega) _ ?_ this
  calc
    _ = ε / 9 * (2 * n + 1) ^ 2 := by simp
    _ ≤ ε / 9 * (2 * n + n) ^ 2 := by gcongr; simp; unfold cornersTheoremBound at hn; omega
    _ = ε * n ^ 2 := by ring
    _ ≤ #A := hAε
    _ = _ := by
      rw [card_image_of_injOn]
      have : Set.InjOn Nat.cast (range n) :=
        (CharP.natCast_injOn_Iio (Fin (2 * n).succ) (2 * n).succ).mono (by simp; omega)
      exact (this.prodMap this).mono hAn


/-- **Roth's theorem** for finite abelian groups.

The maximum density of a 3AP-free set in `G` goes to zero as `|G|` tends to infinity. -/
theorem roth_3ap_theorem (ε : ℝ) (hε : 0 < ε) (hG : cornersTheoremBound ε ≤ card G)
    (A : Finset G) (hAε : ε * card G ≤ #A) : ¬ ThreeAPFree (A : Set G) := by
  /-
    G : Type u_1
    inst✝¹ : AddCommGroup G
    inst✝ : Fintype G
    ε : Real
    hε : LT.lt 0 ε
    hG : LE.le (cornersTheoremBound ε) (Fintype.card G)
    A : Finset G
    hAε : LE.le (HMul.hMul ε ↑(Fintype.card G)) ↑A.card
    ⊢ Not (ThreeAPFree ↑A)
  -/
  rintro hA
  classical
  let B : Finset (G × G) := univ.filter fun (x, y) ↦ y - x ∈ A
  have : ε * card G ^ 2 ≤ #B := by
    calc
      _ = card G * (ε * card G) := by ring
      _ ≤ card G * #A := by gcongr
      _ = #B := ?_
    norm_cast
    rw [← card_univ, ← card_product]
    exact card_equiv ((Equiv.refl _).prodShear fun a ↦ Equiv.addLeft a) (by simp [B])
  obtain ⟨x₁, y₁, x₂, y₂, hx₁y₁, hx₁y₂, hx₂y₁, hxy, hx₁x₂⟩ :
      ∃ x₁ y₁ x₂ y₂, y₁ - x₁ ∈ A ∧ y₂ - x₁ ∈ A ∧ y₁ - x₂ ∈ A ∧ x₁ + y₂ = x₂ + y₁ ∧ x₁ ≠ x₂ := by
    simpa [IsCornerFree, isCorner_iff, B, -exists_and_left, -exists_and_right]
      using corners_theorem ε hε hG B this
  have := hA hx₂y₁ hx₁y₁ hx₁y₂ <| by -- TODO: This really ought to just be `by linear_combination h`
    rw [sub_add_sub_comm, add_comm, add_sub_add_comm, add_right_cancel_iff,
      sub_eq_sub_iff_add_eq_add, add_comm, hxy, add_comm]
  exact hx₁x₂ <| by simpa using this.symm


/-- **Roth's theorem** for `ℕ`.

The maximum density of a 3AP-free set in `{1, ..., n}` goes to zero as `n` tends to infinity. -/
theorem roth_3ap_theorem_nat (ε : ℝ) (hε : 0 < ε) (hG : cornersTheoremBound (ε / 3) ≤ n)
    (A : Finset ℕ) (hAn : A ⊆ range n) (hAε : ε * n ≤ #A) : ¬ ThreeAPFree (A : Set ℕ) := by
  /-
    n : Nat
    ε : Real
    hε : LT.lt 0 ε
    hG : LE.le (cornersTheoremBound (HDiv.hDiv ε 3)) n
    A : Finset Nat
    hAn : HasSubset.Subset A (Finset.range n)
    hAε : LE.le (HMul.hMul ε ↑n) ↑A.card
    ⊢ Not (ThreeAPFree ↑A)
  -/
  rintro hA
  /-
    n : Nat
    ε : Real
    hε : LT.lt 0 ε
    hG : LE.le (cornersTheoremBound (HDiv.hDiv ε 3)) n
    A : Finset Nat
    hAn : HasSubset.Subset A (Finset.range n)
    hAε : LE.le (HMul.hMul ε ↑n) ↑A.card
    hA : ThreeAPFree ↑A
    ⊢ False
  -/
  rw [← coe_subset, coe_range] at hAn
  have : A = Fin.val '' (Nat.cast '' A : Set (Fin (2 * n).succ)) := by
    rw [Set.image_image, Set.image_congr, Set.image_id]
    simp only [mem_coe, Nat.succ_eq_add_one, Fin.val_natCast, id_eq, Nat.mod_succ_eq_iff_lt]
    rintro a ha
    have := hAn ha
    simp at this
    omega
  /-
    n : Nat
    ε : Real
    hε : LT.lt 0 ε
    hG : LE.le (cornersTheoremBound (HDiv.hDiv ε 3)) n
    A : Finset Nat
    hAn : HasSubset.Subset (↑A) (Set.Iio n)
    hAε : LE.le (HMul.hMul ε ↑n) ↑A.card
    hA : ThreeAPFree ↑A
    this : Eq (↑A) (Set.image Fin.val (Set.image Nat.cast ↑A))
    ⊢ False
  -/
  rw [this] at hA
  /-
    n : Nat
    ε : Real
    hε : LT.lt 0 ε
    hG : LE.le (cornersTheoremBound (HDiv.hDiv ε 3)) n
    A : Finset Nat
    hAn : HasSubset.Subset (↑A) (Set.Iio n)
    hAε : LE.le (HMul.hMul ε ↑n) ↑A.card
    hA : ThreeAPFree (Set.image Fin.val (Set.image Nat.cast ↑A))
    this : Eq (↑A) (Set.image Fin.val (Set.image Nat.cast ↑A))
    ⊢ False
  -/
  have := Fin.isAddFreimanIso_Iio two_ne_zero (le_refl (2 * n))
  have := hA.of_image this.isAddFreimanHom Fin.val_injective.injOn <| Set.image_subset_iff.2 <|
      hAn.trans fun x hx ↦ Fin.natCast_strictMono (by omega) <| by
        simpa only [coe_range, Set.mem_Iio] using hx
  /-
    n : Nat
    ε : Real
    hε : LT.lt 0 ε
    hG : LE.le (cornersTheoremBound (HDiv.hDiv ε 3)) n
    A : Finset Nat
    hAn : HasSubset.Subset (↑A) (Set.Iio n)
    hAε : LE.le (HMul.hMul ε ↑n) ↑A.card
    hA : ThreeAPFree (Set.image Fin.val (Set.image Nat.cast ↑A))
    this✝¹ : Eq (↑A) (Set.image Fin.val (Set.image Nat.cast ↑A))
    this✝ : IsAddFreimanIso 2 (Set.Iio ↑n) (Set.Iio n) Fin.val
    this : ThreeAPFree (Set.image (fun x => ↑x) ↑A)
    ⊢ False
  -/
  rw [← coe_image] at this
  /-
    n : Nat
    ε : Real
    hε : LT.lt 0 ε
    hG : LE.le (cornersTheoremBound (HDiv.hDiv ε 3)) n
    A : Finset Nat
    hAn : HasSubset.Subset (↑A) (Set.Iio n)
    hAε : LE.le (HMul.hMul ε ↑n) ↑A.card
    hA : ThreeAPFree (Set.image Fin.val (Set.image Nat.cast ↑A))
    this✝¹ : Eq (↑A) (Set.image Fin.val (Set.image Nat.cast ↑A))
    this✝ : IsAddFreimanIso 2 (Set.Iio ↑n) (Set.Iio n) Fin.val
    this : ThreeAPFree ↑(Finset.image (fun x => ↑x) A)
    ⊢ False
  -/
  refine roth_3ap_theorem (ε / 3) (by positivity) (by simp; omega) _ ?_ this
  calc
    _ = ε / 3 * (2 * n + 1) := by simp
    _ ≤ ε / 3 * (2 * n + n) := by gcongr; simp; unfold cornersTheoremBound at hG; omega
    _ = ε * n := by ring
    _ ≤ #A := hAε
    _ = _ := by
      rw [card_image_of_injOn]
      exact (CharP.natCast_injOn_Iio (Fin (2 * n).succ) (2 * n).succ).mono <| hAn.trans <| by
        simp; omega


/-- **Roth's theorem** for `ℕ` as an asymptotic statement.

The maximum density of a 3AP-free set in `{1, ..., n}` goes to zero as `n` tends to infinity. -/
theorem rothNumberNat_isLittleO_id :
    IsLittleO atTop (fun N ↦ (rothNumberNat N : ℝ)) (fun N ↦ (N : ℝ)) := by
  /-
    ⊢ Asymptotics.IsLittleO Filter.atTop (fun N => ↑(rothNumberNat N)) fun N => ↑N
  -/
  simp only [isLittleO_iff, eventually_atTop, RCLike.norm_natCast]
  /-
    ⊢ ∀ ⦃c : Real⦄, LT.lt 0 c → Exists fun a => ∀ (b : Nat), GE.ge b a → LE.le (↑( …
  -/
  refine fun ε hε ↦ ⟨cornersTheoremBound (ε / 3), fun n hn ↦ ?_⟩
  /-
    ε : Real
    hε : LT.lt 0 ε
    n : Nat
    hn : GE.ge n (cornersTheoremBound (HDiv.hDiv ε 3))
    ⊢ LE.le (↑(rothNumberNat n)) (HMul.hMul ε ↑n)
  -/
  obtain ⟨A, hs₁, hs₂, hs₃⟩ := rothNumberNat_spec n
  /-
    case intro.intro.intro
    ε : Real
    hε : LT.lt 0 ε
    n : Nat
    hn : GE.ge n (cornersTheoremBound (HDiv.hDiv ε 3))
    A : Finset Nat
    hs₁ : HasSubset.Subset A (Finset.range n)
    hs₂ : Eq A.card (rothNumberNat n)
    hs₃ : ThreeAPFree ↑A
    ⊢ LE.le (↑(rothNumberNat n)) (HMul.hMul ε ↑n)
  -/
  rw [← hs₂, ← not_lt]
  /-
    case intro.intro.intro
    ε : Real
    hε : LT.lt 0 ε
    n : Nat
    hn : GE.ge n (cornersTheoremBound (HDiv.hDiv ε 3))
    A : Finset Nat
    hs₁ : HasSubset.Subset A (Finset.range n)
    hs₂ : Eq A.card (rothNumberNat n)
    hs₃ : ThreeAPFree ↑A
    ⊢ Not (LT.lt (HMul.hMul ε ↑n) ↑A.card)
  -/
  exact fun hδn ↦ roth_3ap_theorem_nat ε hε hn _ hs₁ hδn.le hs₃
  /-
    🎉 no goals
  -/

