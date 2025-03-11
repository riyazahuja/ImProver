/-- A partition of `Fin n` into finitely many nonempty subsets, given by the increasing
parameterization of these subsets. We order the subsets by increasing greatest element.
This definition is tailored-made for the Faa di Bruno formula, and probably not useful elsewhere,
because of the specific parameterization by `Fin n` and the peculiar ordering. -/
@[ext]
structure OrderedFinpartition (n : ℕ) where
  /-- The number of parts in the partition -/
  length : ℕ
  /-- The size of each part -/
  partSize : Fin length → ℕ
  partSize_pos : ∀ m, 0 < partSize m
  /-- The increasing parameterization of each part -/
  emb : ∀ m, (Fin (partSize m)) → Fin n
  emb_strictMono : ∀ m, StrictMono (emb m)
  /-- The parts are ordered by increasing greatest element. -/
  parts_strictMono :
    StrictMono fun m ↦ emb m ⟨partSize m - 1, Nat.sub_one_lt_of_lt (partSize_pos m)⟩
  /-- The parts are disjoint -/
  disjoint : PairwiseDisjoint univ fun m ↦ range (emb m)
  /-- The parts cover everything -/
  cover x : ∃ m, x ∈ range (emb m)


/-- The ordered finpartition of `Fin n` into singletons. -/
@[simps] def atomic (n : ℕ) : OrderedFinpartition n where
  length := n
  partSize _ :=  1
  partSize_pos _ := _root_.zero_lt_one
  emb m _ := m
  emb_strictMono _ := Subsingleton.strictMono _
  parts_strictMono := strictMono_id
                           /-
                             𝕜 : Type u_1
                             inst✝⁶ : NontriviallyNormedField 𝕜
                             E : Type u_2
                             inst✝⁵ : NormedAddCommGroup E
                             inst✝⁴ : NormedSpace 𝕜 E
                             F : Type u_3
                             inst✝³ : NormedAddCommGroup F
                             inst✝² : NormedSpace 𝕜 F
                             G : Type u_4
                             inst✝¹ : NormedAddCommGroup G
                             inst✝ : NormedSpace 𝕜 G
                             s : Set E
                             t : Set F
                             q : F → FormalMultilinearSeries 𝕜 F G
                             p : E → FormalMultilinearSeries 𝕜 E F
                             n : Nat
                             x✝³ : Fin n
                             x✝² : Membership.mem Set.univ x✝³
                             x✝¹ : Fin n
                             x✝ : Membership.mem Set.univ x✝¹
                             h : Ne x✝³ x✝¹
                             ⊢ Function.onFun Disjoint (fun m => Set.range ((fun m x => m) m)) x✝³ x✝¹
                           -/
  disjoint _ _ _ _ h := by simpa using h
                           /-
                             🎉 no goals
                           -/
                /-
                  𝕜 : Type u_1
                  inst✝⁶ : NontriviallyNormedField 𝕜
                  E : Type u_2
                  inst✝⁵ : NormedAddCommGroup E
                  inst✝⁴ : NormedSpace 𝕜 E
                  F : Type u_3
                  inst✝³ : NormedAddCommGroup F
                  inst✝² : NormedSpace 𝕜 F
                  G : Type u_4
                  inst✝¹ : NormedAddCommGroup G
                  inst✝ : NormedSpace 𝕜 G
                  s : Set E
                  t : Set F
                  q : F → FormalMultilinearSeries 𝕜 F G
                  p : E → FormalMultilinearSeries 𝕜 E F
                  n : Nat
                  m : Fin n
                  ⊢ Exists fun m_1 => Membership.mem (Set.range ((fun m x => m) m_1)) m
                -/
  cover m := by simp
                /-
                  🎉 no goals
                -/


instance : Inhabited (OrderedFinpartition n) := ⟨atomic n⟩


lemma length_le : c.length ≤ n := by
  /-
    n : Nat
    c : OrderedFinpartition n
    ⊢ LE.le c.length n
  -/
  simpa only [Fintype.card_fin] using Fintype.card_le_of_injective _ c.parts_strictMono.injective
  /-
    🎉 no goals
  -/


lemma partSize_le (m : Fin c.length) : c.partSize m ≤ n := by
  /-
    n : Nat
    c : OrderedFinpartition n
    m : Fin c.length
    ⊢ LE.le (c.partSize m) n
  -/
  simpa only [Fintype.card_fin] using Fintype.card_le_of_injective _ (c.emb_strictMono m).injective
  /-
    🎉 no goals
  -/


/-- Embedding of ordered finpartitions in a sigma type. The sigma type on the right is quite big,
but this is enough to get finiteness of ordered finpartitions. -/
def embSigma (n : ℕ) : OrderedFinpartition n →
    (Σ (l : Fin (n + 1)), Σ (p : Fin l → Fin (n + 1)), Π (i : Fin l), (Fin (p i) → Fin n)) :=
  fun c ↦ ⟨⟨c.length, Order.lt_add_one_iff.mpr c.length_le⟩,
    fun m ↦ ⟨c.partSize m, Order.lt_add_one_iff.mpr (c.partSize_le m)⟩, fun j ↦ c.emb j⟩


lemma injective_embSigma (n : ℕ) : Injective (embSigma n) := by
  /-
    n : Nat
    ⊢ Function.Injective (OrderedFinpartition.embSigma n)
  -/
  rintro ⟨plength, psize, -, pemb, -, -, -, -⟩ ⟨qlength, qsize, -, qemb, -, -, -, -⟩
  /-
    case mk.mk
    n plength : Nat
    psize : Fin plength → Nat
    partSize_pos✝¹ : ∀ (m : Fin plength), LT.lt 0 (psize m)
    pemb : (m : Fin plength) → Fin (psize m) → Fin n
    emb_strictMono✝¹ : ∀ (m : Fin plength), StrictMono (pemb m)
    parts_strictMono✝¹ : StrictMono fun m => pemb m ⟨HSub.hSub (psize m) 1, ⋯⟩
    disjoint✝¹ : Set.univ.PairwiseDisjoint fun m => Set.range (pemb m)
    cover✝¹ : ∀ (x : Fin n), Exists fun m => Membership.mem (Set.range (pemb m)) x
    qlength : Nat
    qsize : Fin qlength → Nat
    partSize_pos✝ : ∀ (m : Fin qlength), LT.lt 0 (qsize m)
    qemb : (m : Fin qlength) → Fin (qsize m) → Fin n
    emb_strictMono✝ : ∀ (m : Fin qlength), StrictMono (qemb m)
    parts_strictMono✝ : StrictMono fun m => qemb m ⟨HSub.hSub (qsize m) 1, ⋯⟩
    disjoint✝ : Set.univ.PairwiseDisjoint fun m => Set.range (qemb m)
    cover✝ : ∀ (x : Fin n), Exists fun m => Membership.mem (Set.range (qemb m)) x
    ⊢ Eq (OrderedFinpartition.embSigma n { length := plength, partSize := psize, p …
  -/
  intro hpq
  /-
    case mk.mk
    n plength : Nat
    psize : Fin plength → Nat
    partSize_pos✝¹ : ∀ (m : Fin plength), LT.lt 0 (psize m)
    pemb : (m : Fin plength) → Fin (psize m) → Fin n
    emb_strictMono✝¹ : ∀ (m : Fin plength), StrictMono (pemb m)
    parts_strictMono✝¹ : StrictMono fun m => pemb m ⟨HSub.hSub (psize m) 1, ⋯⟩
    disjoint✝¹ : Set.univ.PairwiseDisjoint fun m => Set.range (pemb m)
    cover✝¹ : ∀ (x : Fin n), Exists fun m => Membership.mem (Set.range (pemb m)) x
    qlength : Nat
    qsize : Fin qlength → Nat
    partSize_pos✝ : ∀ (m : Fin qlength), LT.lt 0 (qsize m)
    qemb : (m : Fin qlength) → Fin (qsize m) → Fin n
    emb_strictMono✝ : ∀ (m : Fin qlength), StrictMono (qemb m)
    parts_strictMono✝ : StrictMono fun m => qemb m ⟨HSub.hSub (qsize m) 1, ⋯⟩
    disjoint✝ : Set.univ.PairwiseDisjoint fun m => Set.range (qemb m)
    cover✝ : ∀ (x : Fin n), Exists fun m => Membership.mem (Set.range (qemb m)) x
    hpq : Eq (OrderedFinpartition.embSigma n { length := plength, partSize := psiz …
    ⊢ Eq { length := plength, partSize := psize, partSize_pos := partSize_pos✝¹, e …
  -/
  simp_all only [Sigma.mk.inj_iff, heq_eq_eq, true_and, mk.injEq, and_true, Fin.mk.injEq, embSigma]
  /-
    case mk.mk
    n plength : Nat
    psize : Fin plength → Nat
    partSize_pos✝¹ : ∀ (m : Fin plength), LT.lt 0 (psize m)
    pemb : (m : Fin plength) → Fin (psize m) → Fin n
    emb_strictMono✝¹ : ∀ (m : Fin plength), StrictMono (pemb m)
    parts_strictMono✝¹ : StrictMono fun m => pemb m ⟨HSub.hSub (psize m) 1, ⋯⟩
    disjoint✝¹ : Set.univ.PairwiseDisjoint fun m => Set.range (pemb m)
    cover✝¹ : ∀ (x : Fin n), Exists fun m => Membership.mem (Set.range (pemb m)) x
    qlength : Nat
    qsize : Fin qlength → Nat
    partSize_pos✝ : ∀ (m : Fin qlength), LT.lt 0 (qsize m)
    qemb : (m : Fin qlength) → Fin (qsize m) → Fin n
    emb_strictMono✝ : ∀ (m : Fin qlength), StrictMono (qemb m)
    parts_strictMono✝ : StrictMono fun m => qemb m ⟨HSub.hSub (qsize m) 1, ⋯⟩
    disjoint✝ : Set.univ.PairwiseDisjoint fun m => Set.range (qemb m)
    cover✝ : ∀ (x : Fin n), Exists fun m => Membership.mem (Set.range (qemb m)) x
    hpq : And (Eq plength qlength) (HEq ⟨fun m => ⟨psize m, ⋯⟩, fun j => pemb j⟩ ⟨ …
    ⊢ And (HEq psize qsize) (HEq pemb qemb)
  -/
  have : plength = qlength := hpq.1
  /-
    case mk.mk
    n plength : Nat
    psize : Fin plength → Nat
    partSize_pos✝¹ : ∀ (m : Fin plength), LT.lt 0 (psize m)
    pemb : (m : Fin plength) → Fin (psize m) → Fin n
    emb_strictMono✝¹ : ∀ (m : Fin plength), StrictMono (pemb m)
    parts_strictMono✝¹ : StrictMono fun m => pemb m ⟨HSub.hSub (psize m) 1, ⋯⟩
    disjoint✝¹ : Set.univ.PairwiseDisjoint fun m => Set.range (pemb m)
    cover✝¹ : ∀ (x : Fin n), Exists fun m => Membership.mem (Set.range (pemb m)) x
    qlength : Nat
    qsize : Fin qlength → Nat
    partSize_pos✝ : ∀ (m : Fin qlength), LT.lt 0 (qsize m)
    qemb : (m : Fin qlength) → Fin (qsize m) → Fin n
    emb_strictMono✝ : ∀ (m : Fin qlength), StrictMono (qemb m)
    parts_strictMono✝ : StrictMono fun m => qemb m ⟨HSub.hSub (qsize m) 1, ⋯⟩
    disjoint✝ : Set.univ.PairwiseDisjoint fun m => Set.range (qemb m)
    cover✝ : ∀ (x : Fin n), Exists fun m => Membership.mem (Set.range (qemb m)) x
    hpq : And (Eq plength qlength) (HEq ⟨fun m => ⟨psize m, ⋯⟩, fun j => pemb j⟩ ⟨ …
    this : Eq plength qlength
    ⊢ And (HEq psize qsize) (HEq pemb qemb)
  -/
  subst this
  /-
    case mk.mk
    n plength : Nat
    psize : Fin plength → Nat
    partSize_pos✝¹ : ∀ (m : Fin plength), LT.lt 0 (psize m)
    pemb : (m : Fin plength) → Fin (psize m) → Fin n
    emb_strictMono✝¹ : ∀ (m : Fin plength), StrictMono (pemb m)
    parts_strictMono✝¹ : StrictMono fun m => pemb m ⟨HSub.hSub (psize m) 1, ⋯⟩
    disjoint✝¹ : Set.univ.PairwiseDisjoint fun m => Set.range (pemb m)
    cover✝¹ : ∀ (x : Fin n), Exists fun m => Membership.mem (Set.range (pemb m)) x
    qsize : Fin plength → Nat
    partSize_pos✝ : ∀ (m : Fin plength), LT.lt 0 (qsize m)
    qemb : (m : Fin plength) → Fin (qsize m) → Fin n
    emb_strictMono✝ : ∀ (m : Fin plength), StrictMono (qemb m)
    parts_strictMono✝ : StrictMono fun m => qemb m ⟨HSub.hSub (qsize m) 1, ⋯⟩
    disjoint✝ : Set.univ.PairwiseDisjoint fun m => Set.range (qemb m)
    cover✝ : ∀ (x : Fin n), Exists fun m => Membership.mem (Set.range (qemb m)) x
    hpq : And (Eq plength plength) (HEq ⟨fun m => ⟨psize m, ⋯⟩, fun j => pemb j⟩ ⟨ …
    ⊢ And (HEq psize qsize) (HEq pemb qemb)
  -/
  simp_all only [Sigma.mk.inj_iff, heq_eq_eq, true_and, mk.injEq, and_true, Fin.mk.injEq, embSigma]
  /-
    case mk.mk
    n plength : Nat
    psize : Fin plength → Nat
    partSize_pos✝¹ : ∀ (m : Fin plength), LT.lt 0 (psize m)
    pemb : (m : Fin plength) → Fin (psize m) → Fin n
    emb_strictMono✝¹ : ∀ (m : Fin plength), StrictMono (pemb m)
    parts_strictMono✝¹ : StrictMono fun m => pemb m ⟨HSub.hSub (psize m) 1, ⋯⟩
    disjoint✝¹ : Set.univ.PairwiseDisjoint fun m => Set.range (pemb m)
    cover✝¹ : ∀ (x : Fin n), Exists fun m => Membership.mem (Set.range (pemb m)) x
    qsize : Fin plength → Nat
    partSize_pos✝ : ∀ (m : Fin plength), LT.lt 0 (qsize m)
    qemb : (m : Fin plength) → Fin (qsize m) → Fin n
    emb_strictMono✝ : ∀ (m : Fin plength), StrictMono (qemb m)
    parts_strictMono✝ : StrictMono fun m => qemb m ⟨HSub.hSub (qsize m) 1, ⋯⟩
    disjoint✝ : Set.univ.PairwiseDisjoint fun m => Set.range (qemb m)
    cover✝ : ∀ (x : Fin n), Exists fun m => Membership.mem (Set.range (qemb m)) x
    hpq : And (Eq (fun m => ⟨psize m, ⋯⟩) fun m => ⟨qsize m, ⋯⟩) (HEq (fun j => pe …
    ⊢ Eq psize qsize
  -/
  ext i
  /-
    case mk.mk.h
    n plength : Nat
    psize : Fin plength → Nat
    partSize_pos✝¹ : ∀ (m : Fin plength), LT.lt 0 (psize m)
    pemb : (m : Fin plength) → Fin (psize m) → Fin n
    emb_strictMono✝¹ : ∀ (m : Fin plength), StrictMono (pemb m)
    parts_strictMono✝¹ : StrictMono fun m => pemb m ⟨HSub.hSub (psize m) 1, ⋯⟩
    disjoint✝¹ : Set.univ.PairwiseDisjoint fun m => Set.range (pemb m)
    cover✝¹ : ∀ (x : Fin n), Exists fun m => Membership.mem (Set.range (pemb m)) x
    qsize : Fin plength → Nat
    partSize_pos✝ : ∀ (m : Fin plength), LT.lt 0 (qsize m)
    qemb : (m : Fin plength) → Fin (qsize m) → Fin n
    emb_strictMono✝ : ∀ (m : Fin plength), StrictMono (qemb m)
    parts_strictMono✝ : StrictMono fun m => qemb m ⟨HSub.hSub (qsize m) 1, ⋯⟩
    disjoint✝ : Set.univ.PairwiseDisjoint fun m => Set.range (qemb m)
    cover✝ : ∀ (x : Fin n), Exists fun m => Membership.mem (Set.range (qemb m)) x
    hpq : And (Eq (fun m => ⟨psize m, ⋯⟩) fun m => ⟨qsize m, ⋯⟩) (HEq (fun j => pe …
    i : Fin plength
    ⊢ Eq (psize i) (qsize i)
  -/
  exact mk.inj_iff.mp (congr_fun hpq.1 i)
  /-
    🎉 no goals
  -/

/- The best proof would probably to establish the bijection with Finpartitions, but we opt
for a direct argument, embedding `OrderedPartition n` in a type which is obviously finite. -/

noncomputable instance : Fintype (OrderedFinpartition n) :=
  Fintype.ofInjective _ (injective_embSigma n)


instance : Unique (OrderedFinpartition 0) := by
  have : Subsingleton (OrderedFinpartition 0) :=
    Fintype.card_le_one_iff_subsingleton.mp (Fintype.card_le_of_injective _ (injective_embSigma 0))
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    G : Type u_4
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    s : Set E
    t : Set F
    q : F → FormalMultilinearSeries 𝕜 F G
    p : E → FormalMultilinearSeries 𝕜 E F
    n : Nat
    c : OrderedFinpartition n
    this : Subsingleton (OrderedFinpartition 0)
    ⊢ Unique (OrderedFinpartition 0)
  -/
  exact Unique.mk' (OrderedFinpartition 0)
  /-
    🎉 no goals
  -/


lemma exists_inverse {n : ℕ} (c : OrderedFinpartition n) (j : Fin n) :
    ∃ p : Σ m, Fin (c.partSize m), c.emb p.1 p.2 = j := by
  /-
    n : Nat
    c : OrderedFinpartition n
    j : Fin n
    ⊢ Exists fun p => Eq (c.emb p.fst p.snd) j
  -/
  rcases c.cover j with ⟨m, r, hmr⟩
  /-
    case intro.intro
    n : Nat
    c : OrderedFinpartition n
    j : Fin n
    m : Fin c.length
    r : Fin (c.partSize m)
    hmr : Eq (c.emb m r) j
    ⊢ Exists fun p => Eq (c.emb p.fst p.snd) j
  -/
  exact ⟨⟨m, r⟩, hmr⟩
  /-
    🎉 no goals
  -/


lemma emb_injective : Injective (fun (p : Σ m, Fin (c.partSize m)) ↦ c.emb p.1 p.2) := by
  /-
    n : Nat
    c : OrderedFinpartition n
    ⊢ Function.Injective fun p => c.emb p.fst p.snd
  -/
  rintro ⟨m, r⟩ ⟨m', r'⟩ (h : c.emb m r = c.emb m' r')
  have : m = m' := by
    contrapose! h
    have A : Disjoint (range (c.emb m)) (range (c.emb m')) :=
      c.disjoint (mem_univ m) (mem_univ m') h
    apply disjoint_iff_forall_ne.1 A (mem_range_self r) (mem_range_self r')
  /-
    case mk.mk
    n : Nat
    c : OrderedFinpartition n
    m : Fin c.length
    r : Fin (c.partSize m)
    m' : Fin c.length
    r' : Fin (c.partSize m')
    h : Eq (c.emb m r) (c.emb m' r')
    this : Eq m m'
    ⊢ Eq ⟨m, r⟩ ⟨m', r'⟩
  -/
  subst this
  /-
    case mk.mk
    n : Nat
    c : OrderedFinpartition n
    m : Fin c.length
    r r' : Fin (c.partSize m)
    h : Eq (c.emb m r) (c.emb m r')
    ⊢ Eq ⟨m, r⟩ ⟨m, r'⟩
  -/
  simpa using (c.emb_strictMono m).injective h
  /-
    🎉 no goals
  -/


lemma emb_ne_emb_of_ne {i j : Fin c.length} {a : Fin (c.partSize i)} {b : Fin (c.partSize j)}
    (h : i ≠ j) : c.emb i a ≠ c.emb j b :=
                                                       /-
                                                         n : Nat
                                                         c : OrderedFinpartition n
                                                         i j : Fin c.length
                                                         a : Fin (c.partSize i)
                                                         b : Fin (c.partSize j)
                                                         h : Ne i j
                                                         ⊢ Ne ⟨i, a⟩ ⟨j, b⟩
                                                       -/
  c.emb_injective.ne (a₁ := ⟨i, a⟩) (a₂ := ⟨j, b⟩) (by simp [h])
                                                       /-
                                                         🎉 no goals
                                                       -/


/-- Given `j : Fin n`, the index of the part to which it belongs. -/
noncomputable def index (j : Fin n) : Fin c.length :=
  (c.exists_inverse j).choose.1


/-- The inverse of `c.emb` for `c : OrderedFinpartition`. It maps `j : Fin n` to the point in
`Fin (c.partSize (c.index j))` which is mapped back to `j` by `c.emb (c.index j)`. -/
noncomputable def invEmbedding (j : Fin n) :
    Fin (c.partSize (c.index j)) := (c.exists_inverse j).choose.2


@[simp] lemma emb_invEmbedding (j : Fin n) :
    c.emb (c.index j) (c.invEmbedding j) = j :=
  (c.exists_inverse j).choose_spec


/-- An ordered finpartition gives an equivalence between `Fin n` and the disjoint union of the
parts, each of them parameterized by `Fin (c.partSize i)`. -/
noncomputable def equivSigma : ((i : Fin c.length) × Fin (c.partSize i)) ≃ Fin n where
  toFun p := c.emb p.1 p.2
  invFun i := ⟨c.index i, c.invEmbedding i⟩
                    /-
                      𝕜 : Type u_1
                      inst✝⁶ : NontriviallyNormedField 𝕜
                      E : Type u_2
                      inst✝⁵ : NormedAddCommGroup E
                      inst✝⁴ : NormedSpace 𝕜 E
                      F : Type u_3
                      inst✝³ : NormedAddCommGroup F
                      inst✝² : NormedSpace 𝕜 F
                      G : Type u_4
                      inst✝¹ : NormedAddCommGroup G
                      inst✝ : NormedSpace 𝕜 G
                      s : Set E
                      t : Set F
                      q : F → FormalMultilinearSeries 𝕜 F G
                      p : E → FormalMultilinearSeries 𝕜 E F
                      n : Nat
                      c : OrderedFinpartition n
                      x✝ : Fin n
                      ⊢ Eq ((fun p => c.emb p.fst p.snd) ((fun i => ⟨c.index i, c.invEmbedding i⟩) x …
                    -/
                   /-
                     𝕜 : Type u_1
                     inst✝⁶ : NontriviallyNormedField 𝕜
                     E : Type u_2
                     inst✝⁵ : NormedAddCommGroup E
                     inst✝⁴ : NormedSpace 𝕜 E
                     F : Type u_3
                     inst✝³ : NormedAddCommGroup F
                     inst✝² : NormedSpace 𝕜 F
                     G : Type u_4
                     inst✝¹ : NormedAddCommGroup G
                     inst✝ : NormedSpace 𝕜 G
                     s : Set E
                     t : Set F
                     q : F → FormalMultilinearSeries 𝕜 F G
                     p : E → FormalMultilinearSeries 𝕜 E F
                     n : Nat
                     c : OrderedFinpartition n
                     x✝ : Sigma fun i => Fin (c.partSize i)
                     ⊢ Eq ((fun i => ⟨c.index i, c.invEmbedding i⟩) ((fun p => c.emb p.fst p.snd) x …
                   -/
  right_inv _ := by simp
                                          /-
                                            🎉 no goals
                                          -/
                    /-
                      🎉 no goals
                    -/
  left_inv _ := by apply c.emb_injective; simp


@[to_additive] lemma prod_sigma_eq_prod {α : Type*} [CommMonoid α] (v : Fin n → α) :
    ∏ (m : Fin c.length), ∏ (r : Fin (c.partSize m)), v (c.emb m r) = ∏ i, v i := by
  /-
    n : Nat
    c : OrderedFinpartition n
    α : Type u_5
    inst✝ : CommMonoid α
    v : Fin n → α
    ⊢ Eq (Finset.univ.prod fun m => Finset.univ.prod fun r => v (c.emb m r)) (Fins …
  -/
  rw [Finset.prod_sigma']
  /-
    n : Nat
    c : OrderedFinpartition n
    α : Type u_5
    inst✝ : CommMonoid α
    v : Fin n → α
    ⊢ Eq ((Finset.univ.sigma fun m => Finset.univ).prod fun x => v (c.emb x.fst x. …
  -/
  exact Fintype.prod_equiv c.equivSigma _ _ (fun p ↦ rfl)
  /-
    🎉 no goals
  -/


lemma length_pos (h : 0 < n) : 0 < c.length := Nat.zero_lt_of_lt (c.index ⟨0, h⟩).2


lemma neZero_length [NeZero n] (c : OrderedFinpartition n) : NeZero c.length :=
  ⟨(c.length_pos pos').ne'⟩


lemma neZero_partSize (c : OrderedFinpartition n) (i : Fin c.length) : NeZero (c.partSize i) :=
  .of_pos (c.partSize_pos i)


lemma emb_zero [NeZero n] : c.emb (c.index 0) 0 = 0 := by
  /-
    n : Nat
    c : OrderedFinpartition n
    inst✝ : NeZero n
    ⊢ Eq (c.emb (c.index 0) 0) 0
  -/
  apply le_antisymm _ (Fin.zero_le' _)
  /-
    n : Nat
    c : OrderedFinpartition n
    inst✝ : NeZero n
    ⊢ LE.le (c.emb (c.index 0) 0) 0
  -/
  conv_rhs => rw [← c.emb_invEmbedding 0]
  /-
    n : Nat
    c : OrderedFinpartition n
    inst✝ : NeZero n
    ⊢ LE.le (c.emb (c.index 0) 0) (c.emb (c.index 0) (c.invEmbedding 0))
  -/
  apply (c.emb_strictMono _).monotone (Fin.zero_le' _)
  /-
    🎉 no goals
  -/


lemma partSize_eq_one_of_range_emb_eq_singleton
    (c : OrderedFinpartition n) {i : Fin c.length} {j : Fin n}
    (hc : range (c.emb i) = {j}) :
    c.partSize i = 1 := by
  have : Fintype.card (range (c.emb i)) = Fintype.card (Fin (c.partSize i)) :=
    card_range_of_injective (c.emb_strictMono i).injective
  /-
    n : Nat
    c : OrderedFinpartition n
    i : Fin c.length
    j : Fin n
    hc : Eq (Set.range (c.emb i)) (Singleton.singleton j)
    this : Eq (Fintype.card ↑(Set.range (c.emb i))) (Fintype.card (Fin (c.partSize …
    ⊢ Eq (c.partSize i) 1
  -/
  simpa [hc] using this.symm
  /-
    🎉 no goals
  -/


/-- If the left-most part is not `{0}`, then the part containing `0` has at least two elements:
either because it's the left-most part, and then it's not just `0` by assumption, or because it's
not the left-most part and then, by increasingness of maximal elements in parts, it contains
a positive element. -/
lemma one_lt_partSize_index_zero (c : OrderedFinpartition (n + 1)) (hc : range (c.emb 0) ≠ {0}) :
    1 < c.partSize (c.index 0) := by
  have : c.partSize (c.index 0) = Nat.card (range (c.emb (c.index 0))) := by
    rw [Nat.card_range_of_injective (c.emb_strictMono _).injective]; simp
  /-
    n : Nat
    c : OrderedFinpartition (HAdd.hAdd n 1)
    hc : Ne (Set.range (c.emb 0)) (Singleton.singleton 0)
    this : Eq (c.partSize (c.index 0)) (Nat.card ↑(Set.range (c.emb (c.index 0))))
    ⊢ LT.lt 1 (c.partSize (c.index 0))
  -/
  rw [this]
  /-
    n : Nat
    c : OrderedFinpartition (HAdd.hAdd n 1)
    hc : Ne (Set.range (c.emb 0)) (Singleton.singleton 0)
    this : Eq (c.partSize (c.index 0)) (Nat.card ↑(Set.range (c.emb (c.index 0))))
    ⊢ LT.lt 1 (Nat.card ↑(Set.range (c.emb (c.index 0))))
  -/
  rcases eq_or_ne (c.index 0) 0 with h | h
    /-
      case inl
      n : Nat
      c : OrderedFinpartition (HAdd.hAdd n 1)
      hc : Ne (Set.range (c.emb 0)) (Singleton.singleton 0)
      this : Eq (c.partSize (c.index 0)) (Nat.card ↑(Set.range (c.emb (c.index 0))))
      h : Eq (c.index 0) 0
      ⊢ LT.lt 1 (Nat.card ↑(Set.range (c.emb (c.index 0))))
    -/
  · rw [← h] at hc
    have : {0} ⊂ range (c.emb (c.index 0)) := by
      apply ssubset_of_subset_of_ne ?_ hc.symm
      simpa only [singleton_subset_iff, mem_range] using ⟨0, emb_zero c⟩
    /-
      case inl
      n : Nat
      c : OrderedFinpartition (HAdd.hAdd n 1)
      hc : Ne (Set.range (c.emb (c.index 0))) (Singleton.singleton 0)
      this✝ : Eq (c.partSize (c.index 0)) (Nat.card ↑(Set.range (c.emb (c.index 0))))
      h : Eq (c.index 0) 0
      this : HasSSubset.SSubset (Singleton.singleton 0) (Set.range (c.emb (c.index 0 …
      ⊢ LT.lt 1 (Nat.card ↑(Set.range (c.emb (c.index 0))))
    -/
    simpa using Set.Finite.card_lt_card (finite_range _) this
    /-
      🎉 no goals
    -/
    /-
      case inr
      n : Nat
      c : OrderedFinpartition (HAdd.hAdd n 1)
      hc : Ne (Set.range (c.emb 0)) (Singleton.singleton 0)
      this : Eq (c.partSize (c.index 0)) (Nat.card ↑(Set.range (c.emb (c.index 0))))
      h : Ne (c.index 0) 0
      ⊢ LT.lt 1 (Nat.card ↑(Set.range (c.emb (c.index 0))))
    -/
  · apply one_lt_two.trans_le
    have : {c.emb (c.index 0) 0,
        c.emb (c.index 0) ⟨c.partSize (c.index 0) - 1, Nat.sub_one_lt_of_lt (c.partSize_pos _)⟩}
          ⊆ range (c.emb (c.index 0)) := by simp [insert_subset]
    /-
      case inr
      n : Nat
      c : OrderedFinpartition (HAdd.hAdd n 1)
      hc : Ne (Set.range (c.emb 0)) (Singleton.singleton 0)
      this✝ : Eq (c.partSize (c.index 0)) (Nat.card ↑(Set.range (c.emb (c.index 0))))
      h : Ne (c.index 0) 0
      this : HasSubset.Subset (Insert.insert (c.emb (c.index 0) 0) (Singleton.single …
      ⊢ LE.le 2 (Nat.card ↑(Set.range (c.emb (c.index 0))))
    -/
    simp [emb_zero] at this
    /-
      case inr
      n : Nat
      c : OrderedFinpartition (HAdd.hAdd n 1)
      hc : Ne (Set.range (c.emb 0)) (Singleton.singleton 0)
      this✝ : Eq (c.partSize (c.index 0)) (Nat.card ↑(Set.range (c.emb (c.index 0))))
      h : Ne (c.index 0) 0
      this : HasSubset.Subset (Insert.insert 0 (Singleton.singleton (c.emb (c.index  …
      ⊢ LE.le 2 (Nat.card ↑(Set.range (c.emb (c.index 0))))
    -/
    convert Nat.card_mono Subtype.finite this
    /-
      case h.e'_3
      n : Nat
      c : OrderedFinpartition (HAdd.hAdd n 1)
      hc : Ne (Set.range (c.emb 0)) (Singleton.singleton 0)
      this✝ : Eq (c.partSize (c.index 0)) (Nat.card ↑(Set.range (c.emb (c.index 0))))
      h : Ne (c.index 0) 0
      this : HasSubset.Subset (Insert.insert 0 (Singleton.singleton (c.emb (c.index  …
      ⊢ Eq 2 (Nat.card ↑(Insert.insert 0 (Singleton.singleton (c.emb (c.index 0) ⟨HS …
    -/
    simp only [Nat.card_eq_fintype_card, Fintype.card_ofFinset, toFinset_singleton]
    /-
      case h.e'_3
      n : Nat
      c : OrderedFinpartition (HAdd.hAdd n 1)
      hc : Ne (Set.range (c.emb 0)) (Singleton.singleton 0)
      this✝ : Eq (c.partSize (c.index 0)) (Nat.card ↑(Set.range (c.emb (c.index 0))))
      h : Ne (c.index 0) 0
      this : HasSubset.Subset (Insert.insert 0 (Singleton.singleton (c.emb (c.index  …
      ⊢ Eq 2 (Insert.insert 0 (Singleton.singleton (c.emb (c.index 0) ⟨HSub.hSub (c. …
    -/
    apply (Finset.card_pair ?_).symm
    /-
      n : Nat
      c : OrderedFinpartition (HAdd.hAdd n 1)
      hc : Ne (Set.range (c.emb 0)) (Singleton.singleton 0)
      this✝ : Eq (c.partSize (c.index 0)) (Nat.card ↑(Set.range (c.emb (c.index 0))))
      h : Ne (c.index 0) 0
      this : HasSubset.Subset (Insert.insert 0 (Singleton.singleton (c.emb (c.index  …
      ⊢ Ne 0 (c.emb (c.index 0) ⟨HSub.hSub (c.partSize (c.index 0)) 1, ⋯⟩)
    -/
    exact ((Fin.zero_le _).trans_lt (c.parts_strictMono ((pos_iff_ne_zero' (c.index 0)).mpr h))).ne
    /-
      🎉 no goals
    -/


/-- Extend an ordered partition of `n` entries, by adding a new singleton part to the left. -/
def extendLeft (c : OrderedFinpartition n) : OrderedFinpartition (n + 1) where
  length := c.length + 1
  partSize := Fin.cons 1 c.partSize
                                /-
                                  𝕜 : Type u_1
                                  inst✝⁶ : NontriviallyNormedField 𝕜
                                  E : Type u_2
                                  inst✝⁵ : NormedAddCommGroup E
                                  inst✝⁴ : NormedSpace 𝕜 E
                                  F : Type u_3
                                  inst✝³ : NormedAddCommGroup F
                                  inst✝² : NormedSpace 𝕜 F
                                  G : Type u_4
                                  inst✝¹ : NormedAddCommGroup G
                                  inst✝ : NormedSpace 𝕜 G
                                  s : Set E
                                  t : Set F
                                  q : F → FormalMultilinearSeries 𝕜 F G
                                  p : E → FormalMultilinearSeries 𝕜 E F
                                  n : Nat
                                  c✝ c : OrderedFinpartition n
                                  ⊢ LT.lt 0 (Fin.cons 1 c.partSize 0)
                                -/
                                /-
                                  🎉 no goals
                                -/
  partSize_pos := Fin.cases (by simp) (by simp [c.partSize_pos])
                                          /-
                                            🎉 no goals
                                          -/
  emb := Fin.cases (fun _ ↦ 0) (fun m ↦ Fin.succ ∘ c.emb m)
  emb_strictMono := by
    /-
      𝕜 : Type u_1
      inst✝⁶ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace 𝕜 E
      F : Type u_3
      inst✝³ : NormedAddCommGroup F
      inst✝² : NormedSpace 𝕜 F
      G : Type u_4
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace 𝕜 G
      s : Set E
      t : Set F
      q : F → FormalMultilinearSeries 𝕜 F G
      p : E → FormalMultilinearSeries 𝕜 E F
      n : Nat
      c✝ c : OrderedFinpartition n
      ⊢ ∀ (m : Fin (HAdd.hAdd c.length 1)), StrictMono ((fun i => Fin.cases (motive  …
    -/
    refine Fin.cases ?_ (fun i ↦ ?_)
      /-
        case refine_1
        𝕜 : Type u_1
        inst✝⁶ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝⁵ : NormedAddCommGroup E
        inst✝⁴ : NormedSpace 𝕜 E
        F : Type u_3
        inst✝³ : NormedAddCommGroup F
        inst✝² : NormedSpace 𝕜 F
        G : Type u_4
        inst✝¹ : NormedAddCommGroup G
        inst✝ : NormedSpace 𝕜 G
        s : Set E
        t : Set F
        q : F → FormalMultilinearSeries 𝕜 F G
        p : E → FormalMultilinearSeries 𝕜 E F
        n : Nat
        c✝ c : OrderedFinpartition n
        ⊢ StrictMono ((fun i => Fin.cases (motive := fun x => Fin (Fin.cons 1 c.partSi …
      -/
    · exact @Subsingleton.strictMono _ _ _ _ (by simp; infer_instance) _
      /-
        🎉 no goals
      -/
      /-
        case refine_2
        𝕜 : Type u_1
        inst✝⁶ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝⁵ : NormedAddCommGroup E
        inst✝⁴ : NormedSpace 𝕜 E
        F : Type u_3
        inst✝³ : NormedAddCommGroup F
        inst✝² : NormedSpace 𝕜 F
        G : Type u_4
        inst✝¹ : NormedAddCommGroup G
        inst✝ : NormedSpace 𝕜 G
        s : Set E
        t : Set F
        q : F → FormalMultilinearSeries 𝕜 F G
        p : E → FormalMultilinearSeries 𝕜 E F
        n : Nat
        c✝ c : OrderedFinpartition n
        i : Fin c.length
        ⊢ StrictMono ((fun i => Fin.cases (motive := fun x => Fin (Fin.cons 1 c.partSi …
      -/
    · exact strictMono_succ.comp (c.emb_strictMono i)
      /-
        🎉 no goals
      -/
  parts_strictMono i j hij := by
    induction j using Fin.induction with
    | zero => simp at hij
    | succ j => induction i using Fin.induction with
      | zero => simp
      | succ i =>
        simp only [cons_succ, cases_succ, comp_apply, succ_lt_succ_iff]
        exact c.parts_strictMono (by simpa using hij)
  disjoint i hi j hj hij := by
    /-
      𝕜 : Type u_1
      inst✝⁶ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace 𝕜 E
      F : Type u_3
      inst✝³ : NormedAddCommGroup F
      inst✝² : NormedSpace 𝕜 F
      G : Type u_4
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace 𝕜 G
      s : Set E
      t : Set F
      q : F → FormalMultilinearSeries 𝕜 F G
      p : E → FormalMultilinearSeries 𝕜 E F
      n : Nat
      c✝ c : OrderedFinpartition n
      i : Fin (HAdd.hAdd c.length 1)
      hi : Membership.mem Set.univ i
      j : Fin (HAdd.hAdd c.length 1)
      hj : Membership.mem Set.univ j
      hij : Ne i j
      ⊢ Function.onFun Disjoint (fun m => Set.range ((fun i => Fin.cases (motive :=  …
    -/
    wlog h : j < i generalizing i j
    · exact .symm
        (this j (mem_univ j) i (mem_univ i) hij.symm (lt_of_le_of_ne (le_of_not_lt h) hij))
    induction i using Fin.induction with
    | zero => simp at h
    | succ i =>
      induction j using Fin.induction with
      | zero =>
        simp only [onFun, cases_succ, cases_zero]
        apply Set.disjoint_iff_forall_ne.2
        simp only [mem_range, comp_apply, exists_prop', cons_zero, ne_eq, and_imp,
          Nonempty.forall, forall_const, forall_eq', forall_exists_index, forall_apply_eq_imp_iff]
        exact fun _ ↦ succ_ne_zero _
      | succ j =>
        simp only [onFun, cases_succ]
        apply Set.disjoint_iff_forall_ne.2
        simp only [mem_range, comp_apply, ne_eq, forall_exists_index, forall_apply_eq_imp_iff,
          succ_inj]
        intro a b
        apply c.emb_ne_emb_of_ne (by simpa using hij)
  cover := by
    /-
      𝕜 : Type u_1
      inst✝⁶ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace 𝕜 E
      F : Type u_3
      inst✝³ : NormedAddCommGroup F
      inst✝² : NormedSpace 𝕜 F
      G : Type u_4
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace 𝕜 G
      s : Set E
      t : Set F
      q : F → FormalMultilinearSeries 𝕜 F G
      p : E → FormalMultilinearSeries 𝕜 E F
      n : Nat
      c✝ c : OrderedFinpartition n
      ⊢ ∀ (x : Fin (HAdd.hAdd n 1)), Exists fun m => Membership.mem (Set.range ((fun …
    -/
    refine Fin.cases ?_ (fun i ↦ ?_)
      /-
        case refine_1
        𝕜 : Type u_1
        inst✝⁶ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝⁵ : NormedAddCommGroup E
        inst✝⁴ : NormedSpace 𝕜 E
        F : Type u_3
        inst✝³ : NormedAddCommGroup F
        inst✝² : NormedSpace 𝕜 F
        G : Type u_4
        inst✝¹ : NormedAddCommGroup G
        inst✝ : NormedSpace 𝕜 G
        s : Set E
        t : Set F
        q : F → FormalMultilinearSeries 𝕜 F G
        p : E → FormalMultilinearSeries 𝕜 E F
        n : Nat
        c✝ c : OrderedFinpartition n
        ⊢ Exists fun m => Membership.mem (Set.range ((fun i => Fin.cases (motive := fu …
      -/
    · simp only [mem_iUnion, mem_range]
      /-
        case refine_1
        𝕜 : Type u_1
        inst✝⁶ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝⁵ : NormedAddCommGroup E
        inst✝⁴ : NormedSpace 𝕜 E
        F : Type u_3
        inst✝³ : NormedAddCommGroup F
        inst✝² : NormedSpace 𝕜 F
        G : Type u_4
        inst✝¹ : NormedAddCommGroup G
        inst✝ : NormedSpace 𝕜 G
        s : Set E
        t : Set F
        q : F → FormalMultilinearSeries 𝕜 F G
        p : E → FormalMultilinearSeries 𝕜 E F
        n : Nat
        c✝ c : OrderedFinpartition n
        ⊢ Exists fun m => Exists fun y => Eq (Fin.cases (motive := fun x => Fin (Fin.c …
      -/
      exact ⟨0, ⟨0, by simp⟩, by simp⟩
      /-
        🎉 no goals
      -/
      /-
        case refine_2
        𝕜 : Type u_1
        inst✝⁶ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝⁵ : NormedAddCommGroup E
        inst✝⁴ : NormedSpace 𝕜 E
        F : Type u_3
        inst✝³ : NormedAddCommGroup F
        inst✝² : NormedSpace 𝕜 F
        G : Type u_4
        inst✝¹ : NormedAddCommGroup G
        inst✝ : NormedSpace 𝕜 G
        s : Set E
        t : Set F
        q : F → FormalMultilinearSeries 𝕜 F G
        p : E → FormalMultilinearSeries 𝕜 E F
        n : Nat
        c✝ c : OrderedFinpartition n
        i : Fin n
        ⊢ Exists fun m => Membership.mem (Set.range ((fun i => Fin.cases (motive := fu …
      -/
    · simp only [mem_iUnion, mem_range]
      /-
        case refine_2
        𝕜 : Type u_1
        inst✝⁶ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝⁵ : NormedAddCommGroup E
        inst✝⁴ : NormedSpace 𝕜 E
        F : Type u_3
        inst✝³ : NormedAddCommGroup F
        inst✝² : NormedSpace 𝕜 F
        G : Type u_4
        inst✝¹ : NormedAddCommGroup G
        inst✝ : NormedSpace 𝕜 G
        s : Set E
        t : Set F
        q : F → FormalMultilinearSeries 𝕜 F G
        p : E → FormalMultilinearSeries 𝕜 E F
        n : Nat
        c✝ c : OrderedFinpartition n
        i : Fin n
        ⊢ Exists fun m => Exists fun y => Eq (Fin.cases (motive := fun x => Fin (Fin.c …
      -/
      exact ⟨Fin.succ (c.index i), Fin.cast (by simp) (c.invEmbedding i), by simp⟩
      /-
        🎉 no goals
      -/


@[simp] lemma range_extendLeft_zero (c : OrderedFinpartition n) :
    range (c.extendLeft.emb 0) = {0} := by
  /-
    n : Nat
    c : OrderedFinpartition n
    ⊢ Eq (Set.range (c.extendLeft.emb 0)) (Singleton.singleton 0)
  -/
  simp [extendLeft]
  /-
    n : Nat
    c : OrderedFinpartition n
    ⊢ Eq (Set.range fun x => 0) (Singleton.singleton 0)
  -/
  apply @range_const _ _ (by simp; infer_instance)
  /-
    🎉 no goals
  -/


/-- Extend an ordered partition of `n` entries, by adding to the `i`-th part a new point to the
left. -/
def extendMiddle (c : OrderedFinpartition n) (k : Fin c.length) : OrderedFinpartition (n + 1) where
  length := c.length
  partSize := update c.partSize k (c.partSize k + 1)
  partSize_pos m := by
    /-
      𝕜 : Type u_1
      inst✝⁶ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace 𝕜 E
      F : Type u_3
      inst✝³ : NormedAddCommGroup F
      inst✝² : NormedSpace 𝕜 F
      G : Type u_4
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace 𝕜 G
      s : Set E
      t : Set F
      q : F → FormalMultilinearSeries 𝕜 F G
      p : E → FormalMultilinearSeries 𝕜 E F
      n : Nat
      c✝ c : OrderedFinpartition n
      k m : Fin c.length
      ⊢ LT.lt 0 (Function.update c.partSize k (HAdd.hAdd (c.partSize k) 1) m)
    -/
    rcases eq_or_ne m k with rfl | hm
      /-
        case inl
        𝕜 : Type u_1
        inst✝⁶ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝⁵ : NormedAddCommGroup E
        inst✝⁴ : NormedSpace 𝕜 E
        F : Type u_3
        inst✝³ : NormedAddCommGroup F
        inst✝² : NormedSpace 𝕜 F
        G : Type u_4
        inst✝¹ : NormedAddCommGroup G
        inst✝ : NormedSpace 𝕜 G
        s : Set E
        t : Set F
        q : F → FormalMultilinearSeries 𝕜 F G
        p : E → FormalMultilinearSeries 𝕜 E F
        n : Nat
        c✝ c : OrderedFinpartition n
        m : Fin c.length
        ⊢ LT.lt 0 (Function.update c.partSize m (HAdd.hAdd (c.partSize m) 1) m)
      -/
    · simp
      /-
        🎉 no goals
      -/
      /-
        case inr
        𝕜 : Type u_1
        inst✝⁶ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝⁵ : NormedAddCommGroup E
        inst✝⁴ : NormedSpace 𝕜 E
        F : Type u_3
        inst✝³ : NormedAddCommGroup F
        inst✝² : NormedSpace 𝕜 F
        G : Type u_4
        inst✝¹ : NormedAddCommGroup G
        inst✝ : NormedSpace 𝕜 G
        s : Set E
        t : Set F
        q : F → FormalMultilinearSeries 𝕜 F G
        p : E → FormalMultilinearSeries 𝕜 E F
        n : Nat
        c✝ c : OrderedFinpartition n
        k m : Fin c.length
        hm : Ne m k
        ⊢ LT.lt 0 (Function.update c.partSize k (HAdd.hAdd (c.partSize k) 1) m)
      -/
    · simpa [hm] using c.partSize_pos m
      /-
        🎉 no goals
      -/
  emb := by
    /-
      𝕜 : Type u_1
      inst✝⁶ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace 𝕜 E
      F : Type u_3
      inst✝³ : NormedAddCommGroup F
      inst✝² : NormedSpace 𝕜 F
      G : Type u_4
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace 𝕜 G
      s : Set E
      t : Set F
      q : F → FormalMultilinearSeries 𝕜 F G
      p : E → FormalMultilinearSeries 𝕜 E F
      n : Nat
      c✝ c : OrderedFinpartition n
      k : Fin c.length
      ⊢ (m : Fin c.length) → Fin (Function.update c.partSize k (HAdd.hAdd (c.partSiz …
    -/
    intro m
    /-
      𝕜 : Type u_1
      inst✝⁶ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace 𝕜 E
      F : Type u_3
      inst✝³ : NormedAddCommGroup F
      inst✝² : NormedSpace 𝕜 F
      G : Type u_4
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace 𝕜 G
      s : Set E
      t : Set F
      q : F → FormalMultilinearSeries 𝕜 F G
      p : E → FormalMultilinearSeries 𝕜 E F
      n : Nat
      c✝ c : OrderedFinpartition n
      k m : Fin c.length
      ⊢ Fin (Function.update c.partSize k (HAdd.hAdd (c.partSize k) 1) m) → Fin (HAd …
    -/
    by_cases h : m = k
      /-
        case pos
        𝕜 : Type u_1
        inst✝⁶ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝⁵ : NormedAddCommGroup E
        inst✝⁴ : NormedSpace 𝕜 E
        F : Type u_3
        inst✝³ : NormedAddCommGroup F
        inst✝² : NormedSpace 𝕜 F
        G : Type u_4
        inst✝¹ : NormedAddCommGroup G
        inst✝ : NormedSpace 𝕜 G
        s : Set E
        t : Set F
        q : F → FormalMultilinearSeries 𝕜 F G
        p : E → FormalMultilinearSeries 𝕜 E F
        n : Nat
        c✝ c : OrderedFinpartition n
        k m : Fin c.length
        h : Eq m k
        ⊢ Fin (Function.update c.partSize k (HAdd.hAdd (c.partSize k) 1) m) → Fin (HAd …
      -/
    · have : update c.partSize k (c.partSize k + 1) m = c.partSize k + 1 := by rw [h]; simp
      /-
        case pos
        𝕜 : Type u_1
        inst✝⁶ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝⁵ : NormedAddCommGroup E
        inst✝⁴ : NormedSpace 𝕜 E
        F : Type u_3
        inst✝³ : NormedAddCommGroup F
        inst✝² : NormedSpace 𝕜 F
        G : Type u_4
        inst✝¹ : NormedAddCommGroup G
        inst✝ : NormedSpace 𝕜 G
        s : Set E
        t : Set F
        q : F → FormalMultilinearSeries 𝕜 F G
        p : E → FormalMultilinearSeries 𝕜 E F
        n : Nat
        c✝ c : OrderedFinpartition n
        k m : Fin c.length
        h : Eq m k
        this : Eq (Function.update c.partSize k (HAdd.hAdd (c.partSize k) 1) m) (HAdd. …
        ⊢ Fin (Function.update c.partSize k (HAdd.hAdd (c.partSize k) 1) m) → Fin (HAd …
      -/
      exact Fin.cases 0 (succ ∘ c.emb k) ∘ Fin.cast this
      /-
        🎉 no goals
      -/
      /-
        case neg
        𝕜 : Type u_1
        inst✝⁶ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝⁵ : NormedAddCommGroup E
        inst✝⁴ : NormedSpace 𝕜 E
        F : Type u_3
        inst✝³ : NormedAddCommGroup F
        inst✝² : NormedSpace 𝕜 F
        G : Type u_4
        inst✝¹ : NormedAddCommGroup G
        inst✝ : NormedSpace 𝕜 G
        s : Set E
        t : Set F
        q : F → FormalMultilinearSeries 𝕜 F G
        p : E → FormalMultilinearSeries 𝕜 E F
        n : Nat
        c✝ c : OrderedFinpartition n
        k m : Fin c.length
        h : Not (Eq m k)
        ⊢ Fin (Function.update c.partSize k (HAdd.hAdd (c.partSize k) 1) m) → Fin (HAd …
      -/
    · have : update c.partSize k (c.partSize k + 1) m = c.partSize m := by simp [h]
      /-
        case neg
        𝕜 : Type u_1
        inst✝⁶ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝⁵ : NormedAddCommGroup E
        inst✝⁴ : NormedSpace 𝕜 E
        F : Type u_3
        inst✝³ : NormedAddCommGroup F
        inst✝² : NormedSpace 𝕜 F
        G : Type u_4
        inst✝¹ : NormedAddCommGroup G
        inst✝ : NormedSpace 𝕜 G
        s : Set E
        t : Set F
        q : F → FormalMultilinearSeries 𝕜 F G
        p : E → FormalMultilinearSeries 𝕜 E F
        n : Nat
        c✝ c : OrderedFinpartition n
        k m : Fin c.length
        h : Not (Eq m k)
        this : Eq (Function.update c.partSize k (HAdd.hAdd (c.partSize k) 1) m) (c.par …
        ⊢ Fin (Function.update c.partSize k (HAdd.hAdd (c.partSize k) 1) m) → Fin (HAd …
      -/
      exact succ ∘ c.emb m ∘ Fin.cast this
      /-
        🎉 no goals
      -/
  emb_strictMono := by
    /-
      𝕜 : Type u_1
      inst✝⁶ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace 𝕜 E
      F : Type u_3
      inst✝³ : NormedAddCommGroup F
      inst✝² : NormedSpace 𝕜 F
      G : Type u_4
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace 𝕜 G
      s : Set E
      t : Set F
      q : F → FormalMultilinearSeries 𝕜 F G
      p : E → FormalMultilinearSeries 𝕜 E F
      n : Nat
      c✝ c : OrderedFinpartition n
      k : Fin c.length
      ⊢ ∀ (m : Fin c.length), StrictMono (dite (Eq m k) (fun h => letFun ⋯ fun this  …
    -/
    intro m
    /-
      𝕜 : Type u_1
      inst✝⁶ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace 𝕜 E
      F : Type u_3
      inst✝³ : NormedAddCommGroup F
      inst✝² : NormedSpace 𝕜 F
      G : Type u_4
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace 𝕜 G
      s : Set E
      t : Set F
      q : F → FormalMultilinearSeries 𝕜 F G
      p : E → FormalMultilinearSeries 𝕜 E F
      n : Nat
      c✝ c : OrderedFinpartition n
      k m : Fin c.length
      ⊢ StrictMono (dite (Eq m k) (fun h => letFun ⋯ fun this => Function.comp (fun  …
    -/
    rcases eq_or_ne m k with rfl | hm
    · suffices ∀ (a' b' : Fin (c.partSize m + 1)), a' < b' →
          (cases (motive := fun _ ↦ Fin (n + 1)) 0 (succ ∘ c.emb m)) a' <
          (cases (motive := fun _ ↦ Fin (n + 1)) 0 (succ ∘ c.emb m)) b' by
        simp only [↓reduceDIte, comp_apply]
        intro a b hab
        exact this _ _ hab
      /-
        case inl
        𝕜 : Type u_1
        inst✝⁶ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝⁵ : NormedAddCommGroup E
        inst✝⁴ : NormedSpace 𝕜 E
        F : Type u_3
        inst✝³ : NormedAddCommGroup F
        inst✝² : NormedSpace 𝕜 F
        G : Type u_4
        inst✝¹ : NormedAddCommGroup G
        inst✝ : NormedSpace 𝕜 G
        s : Set E
        t : Set F
        q : F → FormalMultilinearSeries 𝕜 F G
        p : E → FormalMultilinearSeries 𝕜 E F
        n : Nat
        c✝ c : OrderedFinpartition n
        m : Fin c.length
        ⊢ ∀ (a' b' : Fin (HAdd.hAdd (c.partSize m) 1)), LT.lt a' b' → LT.lt (Fin.cases …
      -/
      intro a' b' h'
      induction b' using Fin.induction with
      | zero => simp at h'
      | succ b =>
        induction a' using Fin.induction with
        | zero => simp
        | succ a' =>
          simp only [cases_succ, comp_apply, succ_lt_succ_iff]
          exact c.emb_strictMono m (by simpa using h')
      /-
        case inr
        𝕜 : Type u_1
        inst✝⁶ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝⁵ : NormedAddCommGroup E
        inst✝⁴ : NormedSpace 𝕜 E
        F : Type u_3
        inst✝³ : NormedAddCommGroup F
        inst✝² : NormedSpace 𝕜 F
        G : Type u_4
        inst✝¹ : NormedAddCommGroup G
        inst✝ : NormedSpace 𝕜 G
        s : Set E
        t : Set F
        q : F → FormalMultilinearSeries 𝕜 F G
        p : E → FormalMultilinearSeries 𝕜 E F
        n : Nat
        c✝ c : OrderedFinpartition n
        k m : Fin c.length
        hm : Ne m k
        ⊢ StrictMono (dite (Eq m k) (fun h => letFun ⋯ fun this => Function.comp (fun  …
      -/
    · simp only [hm, ↓reduceDIte]
      /-
        case inr
        𝕜 : Type u_1
        inst✝⁶ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝⁵ : NormedAddCommGroup E
        inst✝⁴ : NormedSpace 𝕜 E
        F : Type u_3
        inst✝³ : NormedAddCommGroup F
        inst✝² : NormedSpace 𝕜 F
        G : Type u_4
        inst✝¹ : NormedAddCommGroup G
        inst✝ : NormedSpace 𝕜 G
        s : Set E
        t : Set F
        q : F → FormalMultilinearSeries 𝕜 F G
        p : E → FormalMultilinearSeries 𝕜 E F
        n : Nat
        c✝ c : OrderedFinpartition n
        k m : Fin c.length
        hm : Ne m k
        ⊢ StrictMono (Function.comp Fin.succ (Function.comp (c.emb m) (Fin.cast ⋯)))
      -/
      exact strictMono_succ.comp ((c.emb_strictMono m).comp (by exact fun ⦃a b⦄ h ↦ h))
      /-
        🎉 no goals
      -/
  parts_strictMono := by
    /-
      𝕜 : Type u_1
      inst✝⁶ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace 𝕜 E
      F : Type u_3
      inst✝³ : NormedAddCommGroup F
      inst✝² : NormedSpace 𝕜 F
      G : Type u_4
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace 𝕜 G
      s : Set E
      t : Set F
      q : F → FormalMultilinearSeries 𝕜 F G
      p : E → FormalMultilinearSeries 𝕜 E F
      n : Nat
      c✝ c : OrderedFinpartition n
      k : Fin c.length
      ⊢ StrictMono fun m => dite (Eq m k) (fun h => letFun ⋯ fun this => Function.co …
    -/
    convert strictMono_succ.comp c.parts_strictMono with m
    /-
      case h.e'_5.h
      𝕜 : Type u_1
      inst✝⁶ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace 𝕜 E
      F : Type u_3
      inst✝³ : NormedAddCommGroup F
      inst✝² : NormedSpace 𝕜 F
      G : Type u_4
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace 𝕜 G
      s : Set E
      t : Set F
      q : F → FormalMultilinearSeries 𝕜 F G
      p : E → FormalMultilinearSeries 𝕜 E F
      n : Nat
      c✝ c : OrderedFinpartition n
      k m : Fin c.length
      ⊢ Eq (dite (Eq m k) (fun h => letFun ⋯ fun this => Function.comp (fun i => Fin …
    -/
    rcases eq_or_ne m k with rfl | hm
    · simp only [↓reduceDIte, update_self, add_tsub_cancel_right, comp_apply, cast_mk,
        Nat.succ_eq_add_one]
      /-
        case h.e'_5.h.inl
        𝕜 : Type u_1
        inst✝⁶ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝⁵ : NormedAddCommGroup E
        inst✝⁴ : NormedSpace 𝕜 E
        F : Type u_3
        inst✝³ : NormedAddCommGroup F
        inst✝² : NormedSpace 𝕜 F
        G : Type u_4
        inst✝¹ : NormedAddCommGroup G
        inst✝ : NormedSpace 𝕜 G
        s : Set E
        t : Set F
        q : F → FormalMultilinearSeries 𝕜 F G
        p : E → FormalMultilinearSeries 𝕜 E F
        n : Nat
        c✝ c : OrderedFinpartition n
        m : Fin c.length
        ⊢ Eq (Fin.cases 0 (Function.comp Fin.succ (c.emb m)) ⟨c.partSize m, ⋯⟩) (c.emb …
      -/
      let a : Fin (c.partSize m + 1) := ⟨c.partSize m, lt_add_one (c.partSize m)⟩
      /-
        case h.e'_5.h.inl
        𝕜 : Type u_1
        inst✝⁶ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝⁵ : NormedAddCommGroup E
        inst✝⁴ : NormedSpace 𝕜 E
        F : Type u_3
        inst✝³ : NormedAddCommGroup F
        inst✝² : NormedSpace 𝕜 F
        G : Type u_4
        inst✝¹ : NormedAddCommGroup G
        inst✝ : NormedSpace 𝕜 G
        s : Set E
        t : Set F
        q : F → FormalMultilinearSeries 𝕜 F G
        p : E → FormalMultilinearSeries 𝕜 E F
        n : Nat
        c✝ c : OrderedFinpartition n
        m : Fin c.length
        a : Fin (HAdd.hAdd (c.partSize m) 1) := ⟨c.partSize m, ⋯⟩
        ⊢ Eq (Fin.cases 0 (Function.comp Fin.succ (c.emb m)) ⟨c.partSize m, ⋯⟩) (c.emb …
      -/
      let b : Fin (c.partSize m) := ⟨c.partSize m - 1, Nat.sub_one_lt_of_lt (c.partSize_pos m)⟩
      /-
        case h.e'_5.h.inl
        𝕜 : Type u_1
        inst✝⁶ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝⁵ : NormedAddCommGroup E
        inst✝⁴ : NormedSpace 𝕜 E
        F : Type u_3
        inst✝³ : NormedAddCommGroup F
        inst✝² : NormedSpace 𝕜 F
        G : Type u_4
        inst✝¹ : NormedAddCommGroup G
        inst✝ : NormedSpace 𝕜 G
        s : Set E
        t : Set F
        q : F → FormalMultilinearSeries 𝕜 F G
        p : E → FormalMultilinearSeries 𝕜 E F
        n : Nat
        c✝ c : OrderedFinpartition n
        m : Fin c.length
        a : Fin (HAdd.hAdd (c.partSize m) 1) := ⟨c.partSize m, ⋯⟩
        b : Fin (c.partSize m) := ⟨HSub.hSub (c.partSize m) 1, ⋯⟩
        ⊢ Eq (Fin.cases 0 (Function.comp Fin.succ (c.emb m)) ⟨c.partSize m, ⋯⟩) (c.emb …
      -/
      change (cases (motive := fun _ ↦ Fin (n + 1)) 0 (succ ∘ c.emb m)) a = succ (c.emb m b)
      have : a = succ b := by
        simpa [a, b, succ] using (Nat.sub_eq_iff_eq_add (c.partSize_pos m)).mp rfl
      /-
        case h.e'_5.h.inl
        𝕜 : Type u_1
        inst✝⁶ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝⁵ : NormedAddCommGroup E
        inst✝⁴ : NormedSpace 𝕜 E
        F : Type u_3
        inst✝³ : NormedAddCommGroup F
        inst✝² : NormedSpace 𝕜 F
        G : Type u_4
        inst✝¹ : NormedAddCommGroup G
        inst✝ : NormedSpace 𝕜 G
        s : Set E
        t : Set F
        q : F → FormalMultilinearSeries 𝕜 F G
        p : E → FormalMultilinearSeries 𝕜 E F
        n : Nat
        c✝ c : OrderedFinpartition n
        m : Fin c.length
        a : Fin (HAdd.hAdd (c.partSize m) 1) := ⟨c.partSize m, ⋯⟩
        b : Fin (c.partSize m) := ⟨HSub.hSub (c.partSize m) 1, ⋯⟩
        this : Eq a b.succ
        ⊢ Eq (Fin.cases 0 (Function.comp Fin.succ (c.emb m)) a) (c.emb m b).succ
      -/
      simp [this]
      /-
        🎉 no goals
      -/
      /-
        case h.e'_5.h.inr
        𝕜 : Type u_1
        inst✝⁶ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝⁵ : NormedAddCommGroup E
        inst✝⁴ : NormedSpace 𝕜 E
        F : Type u_3
        inst✝³ : NormedAddCommGroup F
        inst✝² : NormedSpace 𝕜 F
        G : Type u_4
        inst✝¹ : NormedAddCommGroup G
        inst✝ : NormedSpace 𝕜 G
        s : Set E
        t : Set F
        q : F → FormalMultilinearSeries 𝕜 F G
        p : E → FormalMultilinearSeries 𝕜 E F
        n : Nat
        c✝ c : OrderedFinpartition n
        k m : Fin c.length
        hm : Ne m k
        ⊢ Eq (dite (Eq m k) (fun h => letFun ⋯ fun this => Function.comp (fun i => Fin …
      -/
    · simp [hm]
      /-
        🎉 no goals
      -/
  disjoint i hi j hj hij := by
    /-
      𝕜 : Type u_1
      inst✝⁶ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace 𝕜 E
      F : Type u_3
      inst✝³ : NormedAddCommGroup F
      inst✝² : NormedSpace 𝕜 F
      G : Type u_4
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace 𝕜 G
      s : Set E
      t : Set F
      q : F → FormalMultilinearSeries 𝕜 F G
      p : E → FormalMultilinearSeries 𝕜 E F
      n : Nat
      c✝ c : OrderedFinpartition n
      k i : Fin c.length
      hi : Membership.mem Set.univ i
      j : Fin c.length
      hj : Membership.mem Set.univ j
      hij : Ne i j
      ⊢ Function.onFun Disjoint (fun m => Set.range (dite (Eq m k) (fun h => letFun  …
    -/
    wlog h : i ≠ k generalizing i j
    · apply Disjoint.symm
        (this j (mem_univ j) i (mem_univ i) hij.symm ?_)
      /-
        𝕜 : Type u_1
        inst✝⁶ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝⁵ : NormedAddCommGroup E
        inst✝⁴ : NormedSpace 𝕜 E
        F : Type u_3
        inst✝³ : NormedAddCommGroup F
        inst✝² : NormedSpace 𝕜 F
        G : Type u_4
        inst✝¹ : NormedAddCommGroup G
        inst✝ : NormedSpace 𝕜 G
        s : Set E
        t : Set F
        q : F → FormalMultilinearSeries 𝕜 F G
        p : E → FormalMultilinearSeries 𝕜 E F
        n : Nat
        c✝ c : OrderedFinpartition n
        k i : Fin c.length
        hi : Membership.mem Set.univ i
        j : Fin c.length
        hj : Membership.mem Set.univ j
        hij : Ne i j
        this : ∀ (i : Fin c.length), Membership.mem Set.univ i → ∀ (j : Fin c.length), …
        h : Not (Ne i k)
        ⊢ Ne j k
      -/
      simp only [ne_eq, Decidable.not_not] at h
      /-
        𝕜 : Type u_1
        inst✝⁶ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝⁵ : NormedAddCommGroup E
        inst✝⁴ : NormedSpace 𝕜 E
        F : Type u_3
        inst✝³ : NormedAddCommGroup F
        inst✝² : NormedSpace 𝕜 F
        G : Type u_4
        inst✝¹ : NormedAddCommGroup G
        inst✝ : NormedSpace 𝕜 G
        s : Set E
        t : Set F
        q : F → FormalMultilinearSeries 𝕜 F G
        p : E → FormalMultilinearSeries 𝕜 E F
        n : Nat
        c✝ c : OrderedFinpartition n
        k i : Fin c.length
        hi : Membership.mem Set.univ i
        j : Fin c.length
        hj : Membership.mem Set.univ j
        hij : Ne i j
        this : ∀ (i : Fin c.length), Membership.mem Set.univ i → ∀ (j : Fin c.length), …
        h : Eq i k
        ⊢ Ne j k
      -/
      simpa [h] using hij.symm
      /-
        🎉 no goals
      -/
    /-
      𝕜 : Type u_1
      inst✝⁶ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace 𝕜 E
      F : Type u_3
      inst✝³ : NormedAddCommGroup F
      inst✝² : NormedSpace 𝕜 F
      G : Type u_4
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace 𝕜 G
      s : Set E
      t : Set F
      q : F → FormalMultilinearSeries 𝕜 F G
      p : E → FormalMultilinearSeries 𝕜 E F
      n : Nat
      c✝ c : OrderedFinpartition n
      k i : Fin c.length
      hi : Membership.mem Set.univ i
      j : Fin c.length
      hj : Membership.mem Set.univ j
      hij : Ne i j
      h : Ne i k
      ⊢ Function.onFun Disjoint (fun m => Set.range (dite (Eq m k) (fun h => letFun  …
    -/
    rcases eq_or_ne j k with rfl | hj
      /-
        case inl
        𝕜 : Type u_1
        inst✝⁶ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝⁵ : NormedAddCommGroup E
        inst✝⁴ : NormedSpace 𝕜 E
        F : Type u_3
        inst✝³ : NormedAddCommGroup F
        inst✝² : NormedSpace 𝕜 F
        G : Type u_4
        inst✝¹ : NormedAddCommGroup G
        inst✝ : NormedSpace 𝕜 G
        s : Set E
        t : Set F
        q : F → FormalMultilinearSeries 𝕜 F G
        p : E → FormalMultilinearSeries 𝕜 E F
        n : Nat
        c✝ c : OrderedFinpartition n
        i : Fin c.length
        hi : Membership.mem Set.univ i
        j : Fin c.length
        hj : Membership.mem Set.univ j
        hij h : Ne i j
        ⊢ Function.onFun Disjoint (fun m => Set.range (dite (Eq m j) (fun h => letFun  …
      -/
    · simp only [onFun, ↓reduceDIte, Ne.symm hij]
      suffices ∀ (a' : Fin (c.partSize i)) (b' : Fin (c.partSize j + 1)),
          succ (c.emb i a') ≠ cases (motive := fun _ ↦ Fin (n + 1)) 0 (succ ∘ c.emb j) b' by
        apply Set.disjoint_iff_forall_ne.2
        simp only [hij, ↓reduceDIte, mem_range, comp_apply, ne_eq, forall_exists_index,
          forall_apply_eq_imp_iff]
        intro a b
        apply this
      /-
        case inl
        𝕜 : Type u_1
        inst✝⁶ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝⁵ : NormedAddCommGroup E
        inst✝⁴ : NormedSpace 𝕜 E
        F : Type u_3
        inst✝³ : NormedAddCommGroup F
        inst✝² : NormedSpace 𝕜 F
        G : Type u_4
        inst✝¹ : NormedAddCommGroup G
        inst✝ : NormedSpace 𝕜 G
        s : Set E
        t : Set F
        q : F → FormalMultilinearSeries 𝕜 F G
        p : E → FormalMultilinearSeries 𝕜 E F
        n : Nat
        c✝ c : OrderedFinpartition n
        i : Fin c.length
        hi : Membership.mem Set.univ i
        j : Fin c.length
        hj : Membership.mem Set.univ j
        hij h : Ne i j
        ⊢ ∀ (a' : Fin (c.partSize i)) (b' : Fin (HAdd.hAdd (c.partSize j) 1)), Ne (c.e …
      -/
      intro a' b'
      induction b' using Fin.induction with
      | zero => simpa using succ_ne_zero (c.emb i a')
      | succ b' =>
        simp only [Nat.succ_eq_add_one, cases_succ, comp_apply, ne_eq, succ_inj]
        apply c.emb_ne_emb_of_ne hij
      /-
        case inr
        𝕜 : Type u_1
        inst✝⁶ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝⁵ : NormedAddCommGroup E
        inst✝⁴ : NormedSpace 𝕜 E
        F : Type u_3
        inst✝³ : NormedAddCommGroup F
        inst✝² : NormedSpace 𝕜 F
        G : Type u_4
        inst✝¹ : NormedAddCommGroup G
        inst✝ : NormedSpace 𝕜 G
        s : Set E
        t : Set F
        q : F → FormalMultilinearSeries 𝕜 F G
        p : E → FormalMultilinearSeries 𝕜 E F
        n : Nat
        c✝ c : OrderedFinpartition n
        k i : Fin c.length
        hi : Membership.mem Set.univ i
        j : Fin c.length
        hj✝ : Membership.mem Set.univ j
        hij : Ne i j
        h : Ne i k
        hj : Ne j k
        ⊢ Function.onFun Disjoint (fun m => Set.range (dite (Eq m k) (fun h => letFun  …
      -/
    · simp only [onFun, h, ↓reduceDIte, hj]
      /-
        case inr
        𝕜 : Type u_1
        inst✝⁶ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝⁵ : NormedAddCommGroup E
        inst✝⁴ : NormedSpace 𝕜 E
        F : Type u_3
        inst✝³ : NormedAddCommGroup F
        inst✝² : NormedSpace 𝕜 F
        G : Type u_4
        inst✝¹ : NormedAddCommGroup G
        inst✝ : NormedSpace 𝕜 G
        s : Set E
        t : Set F
        q : F → FormalMultilinearSeries 𝕜 F G
        p : E → FormalMultilinearSeries 𝕜 E F
        n : Nat
        c✝ c : OrderedFinpartition n
        k i : Fin c.length
        hi : Membership.mem Set.univ i
        j : Fin c.length
        hj✝ : Membership.mem Set.univ j
        hij : Ne i j
        h : Ne i k
        hj : Ne j k
        ⊢ Disjoint (Set.range (Function.comp Fin.succ (Function.comp (c.emb i) (Fin.ca …
      -/
      apply Set.disjoint_iff_forall_ne.2
      simp only [mem_range, comp_apply, ne_eq, forall_exists_index, forall_apply_eq_imp_iff,
        succ_inj]
      /-
        case inr
        𝕜 : Type u_1
        inst✝⁶ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝⁵ : NormedAddCommGroup E
        inst✝⁴ : NormedSpace 𝕜 E
        F : Type u_3
        inst✝³ : NormedAddCommGroup F
        inst✝² : NormedSpace 𝕜 F
        G : Type u_4
        inst✝¹ : NormedAddCommGroup G
        inst✝ : NormedSpace 𝕜 G
        s : Set E
        t : Set F
        q : F → FormalMultilinearSeries 𝕜 F G
        p : E → FormalMultilinearSeries 𝕜 E F
        n : Nat
        c✝ c : OrderedFinpartition n
        k i : Fin c.length
        hi : Membership.mem Set.univ i
        j : Fin c.length
        hj✝ : Membership.mem Set.univ j
        hij : Ne i j
        h : Ne i k
        hj : Ne j k
        ⊢ ∀ (a : Fin (Function.update c.partSize k (HAdd.hAdd (c.partSize k) 1) i)) (a …
      -/
      intro a b
      /-
        case inr
        𝕜 : Type u_1
        inst✝⁶ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝⁵ : NormedAddCommGroup E
        inst✝⁴ : NormedSpace 𝕜 E
        F : Type u_3
        inst✝³ : NormedAddCommGroup F
        inst✝² : NormedSpace 𝕜 F
        G : Type u_4
        inst✝¹ : NormedAddCommGroup G
        inst✝ : NormedSpace 𝕜 G
        s : Set E
        t : Set F
        q : F → FormalMultilinearSeries 𝕜 F G
        p : E → FormalMultilinearSeries 𝕜 E F
        n : Nat
        c✝ c : OrderedFinpartition n
        k i : Fin c.length
        hi : Membership.mem Set.univ i
        j : Fin c.length
        hj✝ : Membership.mem Set.univ j
        hij : Ne i j
        h : Ne i k
        hj : Ne j k
        a : Fin (Function.update c.partSize k (HAdd.hAdd (c.partSize k) 1) i)
        b : Fin (Function.update c.partSize k (HAdd.hAdd (c.partSize k) 1) j)
        ⊢ Not (Eq (c.emb i (Fin.cast ⋯ a)) (c.emb j (Fin.cast ⋯ b)))
      -/
      apply c.emb_ne_emb_of_ne hij
      /-
        🎉 no goals
      -/
  cover := by
    /-
      𝕜 : Type u_1
      inst✝⁶ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace 𝕜 E
      F : Type u_3
      inst✝³ : NormedAddCommGroup F
      inst✝² : NormedSpace 𝕜 F
      G : Type u_4
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace 𝕜 G
      s : Set E
      t : Set F
      q : F → FormalMultilinearSeries 𝕜 F G
      p : E → FormalMultilinearSeries 𝕜 E F
      n : Nat
      c✝ c : OrderedFinpartition n
      k : Fin c.length
      ⊢ ∀ (x : Fin (HAdd.hAdd n 1)), Exists fun m => Membership.mem (Set.range (dite …
    -/
    refine Fin.cases ?_ (fun i ↦ ?_)
      /-
        case refine_1
        𝕜 : Type u_1
        inst✝⁶ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝⁵ : NormedAddCommGroup E
        inst✝⁴ : NormedSpace 𝕜 E
        F : Type u_3
        inst✝³ : NormedAddCommGroup F
        inst✝² : NormedSpace 𝕜 F
        G : Type u_4
        inst✝¹ : NormedAddCommGroup G
        inst✝ : NormedSpace 𝕜 G
        s : Set E
        t : Set F
        q : F → FormalMultilinearSeries 𝕜 F G
        p : E → FormalMultilinearSeries 𝕜 E F
        n : Nat
        c✝ c : OrderedFinpartition n
        k : Fin c.length
        ⊢ Exists fun m => Membership.mem (Set.range (dite (Eq m k) (fun h => letFun ⋯  …
      -/
    · simp only [mem_iUnion, mem_range]
      /-
        case refine_1
        𝕜 : Type u_1
        inst✝⁶ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝⁵ : NormedAddCommGroup E
        inst✝⁴ : NormedSpace 𝕜 E
        F : Type u_3
        inst✝³ : NormedAddCommGroup F
        inst✝² : NormedSpace 𝕜 F
        G : Type u_4
        inst✝¹ : NormedAddCommGroup G
        inst✝ : NormedSpace 𝕜 G
        s : Set E
        t : Set F
        q : F → FormalMultilinearSeries 𝕜 F G
        p : E → FormalMultilinearSeries 𝕜 E F
        n : Nat
        c✝ c : OrderedFinpartition n
        k : Fin c.length
        ⊢ Exists fun m => Exists fun y => Eq (dite (Eq m k) (fun h => Function.comp (f …
      -/
      exact ⟨k, ⟨0, by simp⟩, by simp⟩
      /-
        🎉 no goals
      -/
      /-
        case refine_2
        𝕜 : Type u_1
        inst✝⁶ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝⁵ : NormedAddCommGroup E
        inst✝⁴ : NormedSpace 𝕜 E
        F : Type u_3
        inst✝³ : NormedAddCommGroup F
        inst✝² : NormedSpace 𝕜 F
        G : Type u_4
        inst✝¹ : NormedAddCommGroup G
        inst✝ : NormedSpace 𝕜 G
        s : Set E
        t : Set F
        q : F → FormalMultilinearSeries 𝕜 F G
        p : E → FormalMultilinearSeries 𝕜 E F
        n : Nat
        c✝ c : OrderedFinpartition n
        k : Fin c.length
        i : Fin n
        ⊢ Exists fun m => Membership.mem (Set.range (dite (Eq m k) (fun h => letFun ⋯  …
      -/
    · simp only [mem_iUnion, mem_range]
      /-
        case refine_2
        𝕜 : Type u_1
        inst✝⁶ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝⁵ : NormedAddCommGroup E
        inst✝⁴ : NormedSpace 𝕜 E
        F : Type u_3
        inst✝³ : NormedAddCommGroup F
        inst✝² : NormedSpace 𝕜 F
        G : Type u_4
        inst✝¹ : NormedAddCommGroup G
        inst✝ : NormedSpace 𝕜 G
        s : Set E
        t : Set F
        q : F → FormalMultilinearSeries 𝕜 F G
        p : E → FormalMultilinearSeries 𝕜 E F
        n : Nat
        c✝ c : OrderedFinpartition n
        k : Fin c.length
        i : Fin n
        ⊢ Exists fun m => Exists fun y => Eq (dite (Eq m k) (fun h => Function.comp (f …
      -/
      rcases eq_or_ne (c.index i) k with rfl | hi
      · have A : update c.partSize (c.index i) (c.partSize (c.index i) + 1) (c.index i) =
          c.partSize (c.index i) + 1 := by simp
        /-
          case refine_2.inl
          𝕜 : Type u_1
          inst✝⁶ : NontriviallyNormedField 𝕜
          E : Type u_2
          inst✝⁵ : NormedAddCommGroup E
          inst✝⁴ : NormedSpace 𝕜 E
          F : Type u_3
          inst✝³ : NormedAddCommGroup F
          inst✝² : NormedSpace 𝕜 F
          G : Type u_4
          inst✝¹ : NormedAddCommGroup G
          inst✝ : NormedSpace 𝕜 G
          s : Set E
          t : Set F
          q : F → FormalMultilinearSeries 𝕜 F G
          p : E → FormalMultilinearSeries 𝕜 E F
          n : Nat
          c✝ c : OrderedFinpartition n
          i : Fin n
          A : Eq (Function.update c.partSize (c.index i) (HAdd.hAdd (c.partSize (c.index …
          ⊢ Exists fun m => Exists fun y => Eq (dite (Eq m (c.index i)) (fun h => Functi …
        -/
        exact ⟨c.index i, cast A.symm (succ (c.invEmbedding i)), by simp⟩
        /-
          🎉 no goals
        -/
      · have A : update c.partSize k (c.partSize k + 1) (c.index i) = c.partSize (c.index i) := by
          simp [hi]
        /-
          case refine_2.inr
          𝕜 : Type u_1
          inst✝⁶ : NontriviallyNormedField 𝕜
          E : Type u_2
          inst✝⁵ : NormedAddCommGroup E
          inst✝⁴ : NormedSpace 𝕜 E
          F : Type u_3
          inst✝³ : NormedAddCommGroup F
          inst✝² : NormedSpace 𝕜 F
          G : Type u_4
          inst✝¹ : NormedAddCommGroup G
          inst✝ : NormedSpace 𝕜 G
          s : Set E
          t : Set F
          q : F → FormalMultilinearSeries 𝕜 F G
          p : E → FormalMultilinearSeries 𝕜 E F
          n : Nat
          c✝ c : OrderedFinpartition n
          k : Fin c.length
          i : Fin n
          hi : Ne (c.index i) k
          A : Eq (Function.update c.partSize k (HAdd.hAdd (c.partSize k) 1) (c.index i)) …
          ⊢ Exists fun m => Exists fun y => Eq (dite (Eq m k) (fun h => Function.comp (f …
        -/
        exact ⟨c.index i, cast A.symm (c.invEmbedding i), by simp [hi]⟩
        /-
          🎉 no goals
        -/


lemma index_extendMiddle_zero (c : OrderedFinpartition n) (i : Fin c.length) :
    (c.extendMiddle i).index 0 = i := by
  /-
    n : Nat
    c : OrderedFinpartition n
    i : Fin c.length
    ⊢ Eq ((c.extendMiddle i).index 0) i
  -/
  have : (c.extendMiddle i).emb i 0 = 0 := by simp [extendMiddle]
  /-
    n : Nat
    c : OrderedFinpartition n
    i : Fin c.length
    this : Eq ((c.extendMiddle i).emb i 0) 0
    ⊢ Eq ((c.extendMiddle i).index 0) i
  -/
  conv_rhs at this => rw [← (c.extendMiddle i).emb_invEmbedding 0]
  /-
    n : Nat
    c : OrderedFinpartition n
    i : Fin c.length
    this : Eq ((c.extendMiddle i).emb i 0) ((c.extendMiddle i).emb ((c.extendMiddl …
    ⊢ Eq ((c.extendMiddle i).index 0) i
  -/
  contrapose! this
  /-
    n : Nat
    c : OrderedFinpartition n
    i : Fin c.length
    this : Not (Eq ((c.extendMiddle i).index 0) i)
    ⊢ Not (Eq ((c.extendMiddle i).emb i 0) ((c.extendMiddle i).emb ((c.extendMiddl …
  -/
  exact (c.extendMiddle i).emb_ne_emb_of_ne (Ne.symm this)
  /-
    🎉 no goals
  -/


lemma range_emb_extendMiddle_ne_singleton_zero (c : OrderedFinpartition n) (i j : Fin c.length) :
    range ((c.extendMiddle i).emb j) ≠ {0} := by
  /-
    n : Nat
    c : OrderedFinpartition n
    i j : Fin c.length
    ⊢ Ne (Set.range ((c.extendMiddle i).emb j)) (Singleton.singleton 0)
  -/
  intro h
  /-
    n : Nat
    c : OrderedFinpartition n
    i j : Fin c.length
    h : Eq (Set.range ((c.extendMiddle i).emb j)) (Singleton.singleton 0)
    ⊢ False
  -/
  rcases eq_or_ne j i with rfl | hij
  · have : Fin.succ (c.emb j 0) ∈ ({0} : Set (Fin n.succ)) := by
      rw [← h]
      simp only [Nat.succ_eq_add_one, mem_range]
      have A : (c.extendMiddle j).partSize j = c.partSize j + 1 := by simp [extendMiddle]
      refine ⟨Fin.cast A.symm (succ 0), ?_⟩
      simp only [extendMiddle, ↓reduceDIte, comp_apply, cast_trans, cast_eq_self, cases_succ]
    /-
      case inl
      n : Nat
      c : OrderedFinpartition n
      j : Fin c.length
      h : Eq (Set.range ((c.extendMiddle j).emb j)) (Singleton.singleton 0)
      this : Membership.mem (Singleton.singleton 0) (c.emb j 0).succ
      ⊢ False
    -/
    simp only [mem_singleton_iff] at this
    /-
      case inl
      n : Nat
      c : OrderedFinpartition n
      j : Fin c.length
      h : Eq (Set.range ((c.extendMiddle j).emb j)) (Singleton.singleton 0)
      this : Eq (c.emb j 0).succ 0
      ⊢ False
    -/
    exact Fin.succ_ne_zero _ this
    /-
      🎉 no goals
    -/
  · have : (c.extendMiddle i).emb j 0 ∈ range ((c.extendMiddle i).emb j) :=
      mem_range_self 0
    /-
      case inr
      n : Nat
      c : OrderedFinpartition n
      i j : Fin c.length
      h : Eq (Set.range ((c.extendMiddle i).emb j)) (Singleton.singleton 0)
      hij : Ne j i
      this : Membership.mem (Set.range ((c.extendMiddle i).emb j)) ((c.extendMiddle  …
      ⊢ False
    -/
    rw [h] at this
    /-
      case inr
      n : Nat
      c : OrderedFinpartition n
      i j : Fin c.length
      h : Eq (Set.range ((c.extendMiddle i).emb j)) (Singleton.singleton 0)
      hij : Ne j i
      this : Membership.mem (Singleton.singleton 0) ((c.extendMiddle i).emb j 0)
      ⊢ False
    -/
    simp only [extendMiddle, hij, ↓reduceDIte, comp_apply, cast_zero, mem_singleton_iff] at this
    /-
      case inr
      n : Nat
      c : OrderedFinpartition n
      i j : Fin c.length
      h : Eq (Set.range ((c.extendMiddle i).emb j)) (Singleton.singleton 0)
      hij : Ne j i
      this : Eq (c.emb j 0).succ 0
      ⊢ False
    -/
    exact Fin.succ_ne_zero _ this
    /-
      🎉 no goals
    -/


/-- Extend an ordered partition of `n` entries, by adding singleton to the left or appending it
to one of the existing part. -/
def extend (c : OrderedFinpartition n) (i : Option (Fin c.length)) : OrderedFinpartition (n + 1) :=
  match i with
  | none => c.extendLeft
  | some i => c.extendMiddle i


/-- Given an ordered finpartition of `n+1`, with a leftmost atom equal to `{0}`, remove this
atom to form an ordered finpartition of `n`. -/
def eraseLeft (c : OrderedFinpartition (n + 1)) (hc : range (c.emb 0) = {0}) :
    OrderedFinpartition n where
  length := c.length - 1
  partSize := by
    /-
      𝕜 : Type u_1
      inst✝⁶ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace 𝕜 E
      F : Type u_3
      inst✝³ : NormedAddCommGroup F
      inst✝² : NormedSpace 𝕜 F
      G : Type u_4
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace 𝕜 G
      s : Set E
      t : Set F
      q : F → FormalMultilinearSeries 𝕜 F G
      p : E → FormalMultilinearSeries 𝕜 E F
      n : Nat
      c✝ : OrderedFinpartition n
      c : OrderedFinpartition (HAdd.hAdd n 1)
      hc : Eq (Set.range (c.emb 0)) (Singleton.singleton 0)
      ⊢ Fin (HSub.hSub c.length 1) → Nat
    -/
    have : c.length - 1 + 1 = c.length := Nat.sub_add_cancel (c.length_pos (Nat.zero_lt_succ n))
    /-
      𝕜 : Type u_1
      inst✝⁶ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace 𝕜 E
      F : Type u_3
      inst✝³ : NormedAddCommGroup F
      inst✝² : NormedSpace 𝕜 F
      G : Type u_4
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace 𝕜 G
      s : Set E
      t : Set F
      q : F → FormalMultilinearSeries 𝕜 F G
      p : E → FormalMultilinearSeries 𝕜 E F
      n : Nat
      c✝ : OrderedFinpartition n
      c : OrderedFinpartition (HAdd.hAdd n 1)
      hc : Eq (Set.range (c.emb 0)) (Singleton.singleton 0)
      this : Eq (HAdd.hAdd (HSub.hSub c.length 1) 1) c.length
      ⊢ Fin (HSub.hSub c.length 1) → Nat
    -/
    exact fun i ↦ c.partSize (Fin.cast this (succ i))
    /-
      🎉 no goals
    -/
  partSize_pos i := c.partSize_pos _
  emb i j := by
    /-
      𝕜 : Type u_1
      inst✝⁶ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace 𝕜 E
      F : Type u_3
      inst✝³ : NormedAddCommGroup F
      inst✝² : NormedSpace 𝕜 F
      G : Type u_4
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace 𝕜 G
      s : Set E
      t : Set F
      q : F → FormalMultilinearSeries 𝕜 F G
      p : E → FormalMultilinearSeries 𝕜 E F
      n : Nat
      c✝ : OrderedFinpartition n
      c : OrderedFinpartition (HAdd.hAdd n 1)
      hc : Eq (Set.range (c.emb 0)) (Singleton.singleton 0)
      i : Fin (HSub.hSub c.length 1)
      j : Fin (letFun ⋯ (fun this i => c.partSize (Fin.cast this i.succ)) i)
      ⊢ Fin n
    -/
    have : c.length - 1 + 1 = c.length := Nat.sub_add_cancel (c.length_pos (Nat.zero_lt_succ n))
    /-
      𝕜 : Type u_1
      inst✝⁶ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace 𝕜 E
      F : Type u_3
      inst✝³ : NormedAddCommGroup F
      inst✝² : NormedSpace 𝕜 F
      G : Type u_4
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace 𝕜 G
      s : Set E
      t : Set F
      q : F → FormalMultilinearSeries 𝕜 F G
      p : E → FormalMultilinearSeries 𝕜 E F
      n : Nat
      c✝ : OrderedFinpartition n
      c : OrderedFinpartition (HAdd.hAdd n 1)
      hc : Eq (Set.range (c.emb 0)) (Singleton.singleton 0)
      i : Fin (HSub.hSub c.length 1)
      j : Fin (letFun ⋯ (fun this i => c.partSize (Fin.cast this i.succ)) i)
      this : Eq (HAdd.hAdd (HSub.hSub c.length 1) 1) c.length
      ⊢ Fin n
    -/
    refine Fin.pred (c.emb (Fin.cast this (succ i)) j) ?_
    /-
      𝕜 : Type u_1
      inst✝⁶ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace 𝕜 E
      F : Type u_3
      inst✝³ : NormedAddCommGroup F
      inst✝² : NormedSpace 𝕜 F
      G : Type u_4
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace 𝕜 G
      s : Set E
      t : Set F
      q : F → FormalMultilinearSeries 𝕜 F G
      p : E → FormalMultilinearSeries 𝕜 E F
      n : Nat
      c✝ : OrderedFinpartition n
      c : OrderedFinpartition (HAdd.hAdd n 1)
      hc : Eq (Set.range (c.emb 0)) (Singleton.singleton 0)
      i : Fin (HSub.hSub c.length 1)
      j : Fin (letFun ⋯ (fun this i => c.partSize (Fin.cast this i.succ)) i)
      this : Eq (HAdd.hAdd (HSub.hSub c.length 1) 1) c.length
      ⊢ Ne (c.emb (Fin.cast this i.succ) j) 0
    -/
    have := c.disjoint (mem_univ (Fin.cast this (succ i))) (mem_univ 0) (ne_of_beq_false rfl)
    /-
      𝕜 : Type u_1
      inst✝⁶ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace 𝕜 E
      F : Type u_3
      inst✝³ : NormedAddCommGroup F
      inst✝² : NormedSpace 𝕜 F
      G : Type u_4
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace 𝕜 G
      s : Set E
      t : Set F
      q : F → FormalMultilinearSeries 𝕜 F G
      p : E → FormalMultilinearSeries 𝕜 E F
      n : Nat
      c✝ : OrderedFinpartition n
      c : OrderedFinpartition (HAdd.hAdd n 1)
      hc : Eq (Set.range (c.emb 0)) (Singleton.singleton 0)
      i : Fin (HSub.hSub c.length 1)
      j : Fin (letFun ⋯ (fun this i => c.partSize (Fin.cast this i.succ)) i)
      this✝ : Eq (HAdd.hAdd (HSub.hSub c.length 1) 1) c.length
      this : Function.onFun Disjoint (fun m => Set.range (c.emb m)) (Fin.cast this✝  …
      ⊢ Ne (c.emb (Fin.cast this✝ i.succ) j) 0
    -/
    exact Set.disjoint_iff_forall_ne.1 this (by simp) (by simp only [mem_singleton_iff, hc])
    /-
      🎉 no goals
    -/
  emb_strictMono i a b hab := by
    /-
      𝕜 : Type u_1
      inst✝⁶ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace 𝕜 E
      F : Type u_3
      inst✝³ : NormedAddCommGroup F
      inst✝² : NormedSpace 𝕜 F
      G : Type u_4
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace 𝕜 G
      s : Set E
      t : Set F
      q : F → FormalMultilinearSeries 𝕜 F G
      p : E → FormalMultilinearSeries 𝕜 E F
      n : Nat
      c✝ : OrderedFinpartition n
      c : OrderedFinpartition (HAdd.hAdd n 1)
      hc : Eq (Set.range (c.emb 0)) (Singleton.singleton 0)
      i : Fin (HSub.hSub c.length 1)
      a b : Fin (letFun ⋯ (fun this i => c.partSize (Fin.cast this i.succ)) i)
      hab : LT.lt a b
      ⊢ LT.lt ((fun i j => letFun ⋯ fun this => (c.emb (Fin.cast this i.succ) j).pre …
    -/
    simp only [pred_lt_pred_iff, Nat.succ_eq_add_one]
    /-
      𝕜 : Type u_1
      inst✝⁶ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace 𝕜 E
      F : Type u_3
      inst✝³ : NormedAddCommGroup F
      inst✝² : NormedSpace 𝕜 F
      G : Type u_4
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace 𝕜 G
      s : Set E
      t : Set F
      q : F → FormalMultilinearSeries 𝕜 F G
      p : E → FormalMultilinearSeries 𝕜 E F
      n : Nat
      c✝ : OrderedFinpartition n
      c : OrderedFinpartition (HAdd.hAdd n 1)
      hc : Eq (Set.range (c.emb 0)) (Singleton.singleton 0)
      i : Fin (HSub.hSub c.length 1)
      a b : Fin (letFun ⋯ (fun this i => c.partSize (Fin.cast this i.succ)) i)
      hab : LT.lt a b
      ⊢ LT.lt (c.emb (Fin.cast ⋯ i.succ) a) (c.emb (Fin.cast ⋯ i.succ) b)
    -/
    apply c.emb_strictMono _ hab
    /-
      🎉 no goals
    -/
  parts_strictMono := by
    /-
      𝕜 : Type u_1
      inst✝⁶ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace 𝕜 E
      F : Type u_3
      inst✝³ : NormedAddCommGroup F
      inst✝² : NormedSpace 𝕜 F
      G : Type u_4
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace 𝕜 G
      s : Set E
      t : Set F
      q : F → FormalMultilinearSeries 𝕜 F G
      p : E → FormalMultilinearSeries 𝕜 E F
      n : Nat
      c✝ : OrderedFinpartition n
      c : OrderedFinpartition (HAdd.hAdd n 1)
      hc : Eq (Set.range (c.emb 0)) (Singleton.singleton 0)
      ⊢ StrictMono fun m => (fun i j => letFun ⋯ fun this => (c.emb (Fin.cast this i …
    -/
    intro i j hij
    /-
      𝕜 : Type u_1
      inst✝⁶ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace 𝕜 E
      F : Type u_3
      inst✝³ : NormedAddCommGroup F
      inst✝² : NormedSpace 𝕜 F
      G : Type u_4
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace 𝕜 G
      s : Set E
      t : Set F
      q : F → FormalMultilinearSeries 𝕜 F G
      p : E → FormalMultilinearSeries 𝕜 E F
      n : Nat
      c✝ : OrderedFinpartition n
      c : OrderedFinpartition (HAdd.hAdd n 1)
      hc : Eq (Set.range (c.emb 0)) (Singleton.singleton 0)
      i j : Fin (HSub.hSub c.length 1)
      hij : LT.lt i j
      ⊢ LT.lt ((fun m => (fun i j => letFun ⋯ fun this => (c.emb (Fin.cast this i.su …
    -/
    simp only [pred_lt_pred_iff, Nat.succ_eq_add_one]
    /-
      𝕜 : Type u_1
      inst✝⁶ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace 𝕜 E
      F : Type u_3
      inst✝³ : NormedAddCommGroup F
      inst✝² : NormedSpace 𝕜 F
      G : Type u_4
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace 𝕜 G
      s : Set E
      t : Set F
      q : F → FormalMultilinearSeries 𝕜 F G
      p : E → FormalMultilinearSeries 𝕜 E F
      n : Nat
      c✝ : OrderedFinpartition n
      c : OrderedFinpartition (HAdd.hAdd n 1)
      hc : Eq (Set.range (c.emb 0)) (Singleton.singleton 0)
      i j : Fin (HSub.hSub c.length 1)
      hij : LT.lt i j
      ⊢ LT.lt (c.emb (Fin.cast ⋯ i.succ) ⟨HSub.hSub (c.partSize (Fin.cast ⋯ i.succ)) …
    -/
    apply c.parts_strictMono (cast_strictMono _ (strictMono_succ hij))
    /-
      🎉 no goals
    -/
  disjoint i _ j _ hij := by
    /-
      𝕜 : Type u_1
      inst✝⁶ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace 𝕜 E
      F : Type u_3
      inst✝³ : NormedAddCommGroup F
      inst✝² : NormedSpace 𝕜 F
      G : Type u_4
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace 𝕜 G
      s : Set E
      t : Set F
      q : F → FormalMultilinearSeries 𝕜 F G
      p : E → FormalMultilinearSeries 𝕜 E F
      n : Nat
      c✝ : OrderedFinpartition n
      c : OrderedFinpartition (HAdd.hAdd n 1)
      hc : Eq (Set.range (c.emb 0)) (Singleton.singleton 0)
      i : Fin (HSub.hSub c.length 1)
      x✝¹ : Membership.mem Set.univ i
      j : Fin (HSub.hSub c.length 1)
      x✝ : Membership.mem Set.univ j
      hij : Ne i j
      ⊢ Function.onFun Disjoint (fun m => Set.range ((fun i j => letFun ⋯ fun this = …
    -/
    apply Set.disjoint_iff_forall_ne.2
    /-
      𝕜 : Type u_1
      inst✝⁶ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace 𝕜 E
      F : Type u_3
      inst✝³ : NormedAddCommGroup F
      inst✝² : NormedSpace 𝕜 F
      G : Type u_4
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace 𝕜 G
      s : Set E
      t : Set F
      q : F → FormalMultilinearSeries 𝕜 F G
      p : E → FormalMultilinearSeries 𝕜 E F
      n : Nat
      c✝ : OrderedFinpartition n
      c : OrderedFinpartition (HAdd.hAdd n 1)
      hc : Eq (Set.range (c.emb 0)) (Singleton.singleton 0)
      i : Fin (HSub.hSub c.length 1)
      x✝¹ : Membership.mem Set.univ i
      j : Fin (HSub.hSub c.length 1)
      x✝ : Membership.mem Set.univ j
      hij : Ne i j
      ⊢ ∀ ⦃a : Fin n⦄, Membership.mem ((fun m => Set.range ((fun i j => letFun ⋯ fun …
    -/
    simp only [mem_range, ne_eq, forall_exists_index, forall_apply_eq_imp_iff, pred_inj]
    /-
      𝕜 : Type u_1
      inst✝⁶ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace 𝕜 E
      F : Type u_3
      inst✝³ : NormedAddCommGroup F
      inst✝² : NormedSpace 𝕜 F
      G : Type u_4
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace 𝕜 G
      s : Set E
      t : Set F
      q : F → FormalMultilinearSeries 𝕜 F G
      p : E → FormalMultilinearSeries 𝕜 E F
      n : Nat
      c✝ : OrderedFinpartition n
      c : OrderedFinpartition (HAdd.hAdd n 1)
      hc : Eq (Set.range (c.emb 0)) (Singleton.singleton 0)
      i : Fin (HSub.hSub c.length 1)
      x✝¹ : Membership.mem Set.univ i
      j : Fin (HSub.hSub c.length 1)
      x✝ : Membership.mem Set.univ j
      hij : Ne i j
      ⊢ ∀ (a : Fin (c.partSize (Fin.cast ⋯ i.succ))) (a_1 : Fin (c.partSize (Fin.cas …
    -/
    intro a b
    /-
      𝕜 : Type u_1
      inst✝⁶ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace 𝕜 E
      F : Type u_3
      inst✝³ : NormedAddCommGroup F
      inst✝² : NormedSpace 𝕜 F
      G : Type u_4
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace 𝕜 G
      s : Set E
      t : Set F
      q : F → FormalMultilinearSeries 𝕜 F G
      p : E → FormalMultilinearSeries 𝕜 E F
      n : Nat
      c✝ : OrderedFinpartition n
      c : OrderedFinpartition (HAdd.hAdd n 1)
      hc : Eq (Set.range (c.emb 0)) (Singleton.singleton 0)
      i : Fin (HSub.hSub c.length 1)
      x✝¹ : Membership.mem Set.univ i
      j : Fin (HSub.hSub c.length 1)
      x✝ : Membership.mem Set.univ j
      hij : Ne i j
      a : Fin (c.partSize (Fin.cast ⋯ i.succ))
      b : Fin (c.partSize (Fin.cast ⋯ j.succ))
      ⊢ Not (Eq (c.emb (Fin.cast ⋯ i.succ) a) (c.emb (Fin.cast ⋯ j.succ) b))
    -/
    exact c.emb_ne_emb_of_ne ((cast_injective _).ne (by simpa using hij))
    /-
      🎉 no goals
    -/
  cover x := by
    /-
      𝕜 : Type u_1
      inst✝⁶ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace 𝕜 E
      F : Type u_3
      inst✝³ : NormedAddCommGroup F
      inst✝² : NormedSpace 𝕜 F
      G : Type u_4
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace 𝕜 G
      s : Set E
      t : Set F
      q : F → FormalMultilinearSeries 𝕜 F G
      p : E → FormalMultilinearSeries 𝕜 E F
      n : Nat
      c✝ : OrderedFinpartition n
      c : OrderedFinpartition (HAdd.hAdd n 1)
      hc : Eq (Set.range (c.emb 0)) (Singleton.singleton 0)
      x : Fin n
      ⊢ Exists fun m => Membership.mem (Set.range ((fun i j => letFun ⋯ fun this =>  …
    -/
    simp only [mem_iUnion, mem_range]
    obtain ⟨i, j, hij⟩ : ∃ (i : Fin c.length), ∃ (j : Fin (c.partSize i)), c.emb i j = succ x :=
      ⟨c.index (succ x), c.invEmbedding (succ x), by simp⟩
    have A : c.length = c.length - 1 + 1 :=
      (Nat.sub_add_cancel (c.length_pos (Nat.zero_lt_succ n))).symm
    have i_ne : i ≠ 0 := by
      intro h
      have : succ x ∈ range (c.emb i) := by rw [← hij]; apply mem_range_self
      rw [h, hc, mem_singleton_iff] at this
      exact Fin.succ_ne_zero _ this
    /-
      case intro.intro
      𝕜 : Type u_1
      inst✝⁶ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace 𝕜 E
      F : Type u_3
      inst✝³ : NormedAddCommGroup F
      inst✝² : NormedSpace 𝕜 F
      G : Type u_4
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace 𝕜 G
      s : Set E
      t : Set F
      q : F → FormalMultilinearSeries 𝕜 F G
      p : E → FormalMultilinearSeries 𝕜 E F
      n : Nat
      c✝ : OrderedFinpartition n
      c : OrderedFinpartition (HAdd.hAdd n 1)
      hc : Eq (Set.range (c.emb 0)) (Singleton.singleton 0)
      x : Fin n
      i : Fin c.length
      j : Fin (c.partSize i)
      hij : Eq (c.emb i j) x.succ
      A : Eq c.length (HAdd.hAdd (HSub.hSub c.length 1) 1)
      i_ne : Ne i 0
      ⊢ Exists fun m => Exists fun y => Eq ((c.emb (Fin.cast ⋯ m.succ) y).pred ⋯) x
    -/
    refine ⟨pred (Fin.cast A i) (by simpa using i_ne), Fin.cast (by simp) j, ?_⟩
    /-
      case intro.intro
      𝕜 : Type u_1
      inst✝⁶ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace 𝕜 E
      F : Type u_3
      inst✝³ : NormedAddCommGroup F
      inst✝² : NormedSpace 𝕜 F
      G : Type u_4
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace 𝕜 G
      s : Set E
      t : Set F
      q : F → FormalMultilinearSeries 𝕜 F G
      p : E → FormalMultilinearSeries 𝕜 E F
      n : Nat
      c✝ : OrderedFinpartition n
      c : OrderedFinpartition (HAdd.hAdd n 1)
      hc : Eq (Set.range (c.emb 0)) (Singleton.singleton 0)
      x : Fin n
      i : Fin c.length
      j : Fin (c.partSize i)
      hij : Eq (c.emb i j) x.succ
      A : Eq c.length (HAdd.hAdd (HSub.hSub c.length 1) 1)
      i_ne : Ne i 0
      ⊢ Eq ((c.emb (Fin.cast ⋯ ((Fin.cast A i).pred ⋯).succ) (Fin.cast ⋯ j)).pred ⋯) x
    -/
    have : x = pred (succ x) (succ_ne_zero x) := rfl
    /-
      case intro.intro
      𝕜 : Type u_1
      inst✝⁶ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace 𝕜 E
      F : Type u_3
      inst✝³ : NormedAddCommGroup F
      inst✝² : NormedSpace 𝕜 F
      G : Type u_4
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace 𝕜 G
      s : Set E
      t : Set F
      q : F → FormalMultilinearSeries 𝕜 F G
      p : E → FormalMultilinearSeries 𝕜 E F
      n : Nat
      c✝ : OrderedFinpartition n
      c : OrderedFinpartition (HAdd.hAdd n 1)
      hc : Eq (Set.range (c.emb 0)) (Singleton.singleton 0)
      x : Fin n
      i : Fin c.length
      j : Fin (c.partSize i)
      hij : Eq (c.emb i j) x.succ
      A : Eq c.length (HAdd.hAdd (HSub.hSub c.length 1) 1)
      i_ne : Ne i 0
      this : Eq x (x.succ.pred ⋯)
      ⊢ Eq ((c.emb (Fin.cast ⋯ ((Fin.cast A i).pred ⋯).succ) (Fin.cast ⋯ j)).pred ⋯) x
    -/
    rw [this]
    /-
      case intro.intro
      𝕜 : Type u_1
      inst✝⁶ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace 𝕜 E
      F : Type u_3
      inst✝³ : NormedAddCommGroup F
      inst✝² : NormedSpace 𝕜 F
      G : Type u_4
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace 𝕜 G
      s : Set E
      t : Set F
      q : F → FormalMultilinearSeries 𝕜 F G
      p : E → FormalMultilinearSeries 𝕜 E F
      n : Nat
      c✝ : OrderedFinpartition n
      c : OrderedFinpartition (HAdd.hAdd n 1)
      hc : Eq (Set.range (c.emb 0)) (Singleton.singleton 0)
      x : Fin n
      i : Fin c.length
      j : Fin (c.partSize i)
      hij : Eq (c.emb i j) x.succ
      A : Eq c.length (HAdd.hAdd (HSub.hSub c.length 1) 1)
      i_ne : Ne i 0
      this : Eq x (x.succ.pred ⋯)
      ⊢ Eq ((c.emb (Fin.cast ⋯ ((Fin.cast A i).pred ⋯).succ) (Fin.cast ⋯ j)).pred ⋯) …
    -/
    congr
    /-
      case intro.intro.e_i
      𝕜 : Type u_1
      inst✝⁶ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace 𝕜 E
      F : Type u_3
      inst✝³ : NormedAddCommGroup F
      inst✝² : NormedSpace 𝕜 F
      G : Type u_4
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace 𝕜 G
      s : Set E
      t : Set F
      q : F → FormalMultilinearSeries 𝕜 F G
      p : E → FormalMultilinearSeries 𝕜 E F
      n : Nat
      c✝ : OrderedFinpartition n
      c : OrderedFinpartition (HAdd.hAdd n 1)
      hc : Eq (Set.range (c.emb 0)) (Singleton.singleton 0)
      x : Fin n
      i : Fin c.length
      j : Fin (c.partSize i)
      hij : Eq (c.emb i j) x.succ
      A : Eq c.length (HAdd.hAdd (HSub.hSub c.length 1) 1)
      i_ne : Ne i 0
      this : Eq x (x.succ.pred ⋯)
      ⊢ Eq (c.emb (Fin.cast ⋯ ((Fin.cast A i).pred ⋯).succ) (Fin.cast ⋯ j)) x.succ
    -/
    rw [← hij]
    /-
      case intro.intro.e_i
      𝕜 : Type u_1
      inst✝⁶ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace 𝕜 E
      F : Type u_3
      inst✝³ : NormedAddCommGroup F
      inst✝² : NormedSpace 𝕜 F
      G : Type u_4
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace 𝕜 G
      s : Set E
      t : Set F
      q : F → FormalMultilinearSeries 𝕜 F G
      p : E → FormalMultilinearSeries 𝕜 E F
      n : Nat
      c✝ : OrderedFinpartition n
      c : OrderedFinpartition (HAdd.hAdd n 1)
      hc : Eq (Set.range (c.emb 0)) (Singleton.singleton 0)
      x : Fin n
      i : Fin c.length
      j : Fin (c.partSize i)
      hij : Eq (c.emb i j) x.succ
      A : Eq c.length (HAdd.hAdd (HSub.hSub c.length 1) 1)
      i_ne : Ne i 0
      this : Eq x (x.succ.pred ⋯)
      ⊢ Eq (c.emb (Fin.cast ⋯ ((Fin.cast A i).pred ⋯).succ) (Fin.cast ⋯ j)) (c.emb i …
    -/
    congr 1
      /-
        case intro.intro.e_i.h.e_3.h
        𝕜 : Type u_1
        inst✝⁶ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝⁵ : NormedAddCommGroup E
        inst✝⁴ : NormedSpace 𝕜 E
        F : Type u_3
        inst✝³ : NormedAddCommGroup F
        inst✝² : NormedSpace 𝕜 F
        G : Type u_4
        inst✝¹ : NormedAddCommGroup G
        inst✝ : NormedSpace 𝕜 G
        s : Set E
        t : Set F
        q : F → FormalMultilinearSeries 𝕜 F G
        p : E → FormalMultilinearSeries 𝕜 E F
        n : Nat
        c✝ : OrderedFinpartition n
        c : OrderedFinpartition (HAdd.hAdd n 1)
        hc : Eq (Set.range (c.emb 0)) (Singleton.singleton 0)
        x : Fin n
        i : Fin c.length
        j : Fin (c.partSize i)
        hij : Eq (c.emb i j) x.succ
        A : Eq c.length (HAdd.hAdd (HSub.hSub c.length 1) 1)
        i_ne : Ne i 0
        this : Eq x (x.succ.pred ⋯)
        ⊢ Eq (Fin.cast ⋯ ((Fin.cast A i).pred ⋯).succ) i
      -/
    · simp
      /-
        🎉 no goals
      -/
      /-
        case intro.intro.e_i.h.e_4
        𝕜 : Type u_1
        inst✝⁶ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝⁵ : NormedAddCommGroup E
        inst✝⁴ : NormedSpace 𝕜 E
        F : Type u_3
        inst✝³ : NormedAddCommGroup F
        inst✝² : NormedSpace 𝕜 F
        G : Type u_4
        inst✝¹ : NormedAddCommGroup G
        inst✝ : NormedSpace 𝕜 G
        s : Set E
        t : Set F
        q : F → FormalMultilinearSeries 𝕜 F G
        p : E → FormalMultilinearSeries 𝕜 E F
        n : Nat
        c✝ : OrderedFinpartition n
        c : OrderedFinpartition (HAdd.hAdd n 1)
        hc : Eq (Set.range (c.emb 0)) (Singleton.singleton 0)
        x : Fin n
        i : Fin c.length
        j : Fin (c.partSize i)
        hij : Eq (c.emb i j) x.succ
        A : Eq c.length (HAdd.hAdd (HSub.hSub c.length 1) 1)
        i_ne : Ne i 0
        this : Eq x (x.succ.pred ⋯)
        ⊢ HEq (Fin.cast ⋯ j) j
      -/
    · simp [Fin.heq_ext_iff]
      /-
        🎉 no goals
      -/


/-- Given an ordered finpartition of `n+1`, with a leftmost atom different from `{0}`, remove `{0}`
from the atom that contains it, to form an ordered finpartition of `n`. -/
def eraseMiddle (c : OrderedFinpartition (n + 1)) (hc : range (c.emb 0) ≠ {0}) :
    OrderedFinpartition n where
  length := c.length
  partSize := update c.partSize (c.index 0) (c.partSize (c.index 0) - 1)
  partSize_pos i := by
    /-
      𝕜 : Type u_1
      inst✝⁶ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace 𝕜 E
      F : Type u_3
      inst✝³ : NormedAddCommGroup F
      inst✝² : NormedSpace 𝕜 F
      G : Type u_4
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace 𝕜 G
      s : Set E
      t : Set F
      q : F → FormalMultilinearSeries 𝕜 F G
      p : E → FormalMultilinearSeries 𝕜 E F
      n : Nat
      c✝ : OrderedFinpartition n
      c : OrderedFinpartition (HAdd.hAdd n 1)
      hc : Ne (Set.range (c.emb 0)) (Singleton.singleton 0)
      i : Fin c.length
      ⊢ LT.lt 0 (Function.update c.partSize (c.index 0) (HSub.hSub (c.partSize (c.in …
    -/
    rcases eq_or_ne i (c.index 0) with rfl | hi
      /-
        case inl
        𝕜 : Type u_1
        inst✝⁶ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝⁵ : NormedAddCommGroup E
        inst✝⁴ : NormedSpace 𝕜 E
        F : Type u_3
        inst✝³ : NormedAddCommGroup F
        inst✝² : NormedSpace 𝕜 F
        G : Type u_4
        inst✝¹ : NormedAddCommGroup G
        inst✝ : NormedSpace 𝕜 G
        s : Set E
        t : Set F
        q : F → FormalMultilinearSeries 𝕜 F G
        p : E → FormalMultilinearSeries 𝕜 E F
        n : Nat
        c✝ : OrderedFinpartition n
        c : OrderedFinpartition (HAdd.hAdd n 1)
        hc : Ne (Set.range (c.emb 0)) (Singleton.singleton 0)
        ⊢ LT.lt 0 (Function.update c.partSize (c.index 0) (HSub.hSub (c.partSize (c.in …
      -/
    · simpa using c.one_lt_partSize_index_zero hc
      /-
        🎉 no goals
      -/
      /-
        case inr
        𝕜 : Type u_1
        inst✝⁶ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝⁵ : NormedAddCommGroup E
        inst✝⁴ : NormedSpace 𝕜 E
        F : Type u_3
        inst✝³ : NormedAddCommGroup F
        inst✝² : NormedSpace 𝕜 F
        G : Type u_4
        inst✝¹ : NormedAddCommGroup G
        inst✝ : NormedSpace 𝕜 G
        s : Set E
        t : Set F
        q : F → FormalMultilinearSeries 𝕜 F G
        p : E → FormalMultilinearSeries 𝕜 E F
        n : Nat
        c✝ : OrderedFinpartition n
        c : OrderedFinpartition (HAdd.hAdd n 1)
        hc : Ne (Set.range (c.emb 0)) (Singleton.singleton 0)
        i : Fin c.length
        hi : Ne i (c.index 0)
        ⊢ LT.lt 0 (Function.update c.partSize (c.index 0) (HSub.hSub (c.partSize (c.in …
      -/
    · simp only [ne_eq, hi, not_false_eq_true, update_of_ne]
      /-
        case inr
        𝕜 : Type u_1
        inst✝⁶ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝⁵ : NormedAddCommGroup E
        inst✝⁴ : NormedSpace 𝕜 E
        F : Type u_3
        inst✝³ : NormedAddCommGroup F
        inst✝² : NormedSpace 𝕜 F
        G : Type u_4
        inst✝¹ : NormedAddCommGroup G
        inst✝ : NormedSpace 𝕜 G
        s : Set E
        t : Set F
        q : F → FormalMultilinearSeries 𝕜 F G
        p : E → FormalMultilinearSeries 𝕜 E F
        n : Nat
        c✝ : OrderedFinpartition n
        c : OrderedFinpartition (HAdd.hAdd n 1)
        hc : Ne (Set.range (c.emb 0)) (Singleton.singleton 0)
        i : Fin c.length
        hi : Ne i (c.index 0)
        ⊢ LT.lt 0 (c.partSize i)
      -/
      exact c.partSize_pos i
      /-
        🎉 no goals
      -/
  emb i j := by
    /-
      𝕜 : Type u_1
      inst✝⁶ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace 𝕜 E
      F : Type u_3
      inst✝³ : NormedAddCommGroup F
      inst✝² : NormedSpace 𝕜 F
      G : Type u_4
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace 𝕜 G
      s : Set E
      t : Set F
      q : F → FormalMultilinearSeries 𝕜 F G
      p : E → FormalMultilinearSeries 𝕜 E F
      n : Nat
      c✝ : OrderedFinpartition n
      c : OrderedFinpartition (HAdd.hAdd n 1)
      hc : Ne (Set.range (c.emb 0)) (Singleton.singleton 0)
      i : Fin c.length
      j : Fin (Function.update c.partSize (c.index 0) (HSub.hSub (c.partSize (c.inde …
      ⊢ Fin n
    -/
    by_cases h : i = c.index 0
      /-
        case pos
        𝕜 : Type u_1
        inst✝⁶ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝⁵ : NormedAddCommGroup E
        inst✝⁴ : NormedSpace 𝕜 E
        F : Type u_3
        inst✝³ : NormedAddCommGroup F
        inst✝² : NormedSpace 𝕜 F
        G : Type u_4
        inst✝¹ : NormedAddCommGroup G
        inst✝ : NormedSpace 𝕜 G
        s : Set E
        t : Set F
        q : F → FormalMultilinearSeries 𝕜 F G
        p : E → FormalMultilinearSeries 𝕜 E F
        n : Nat
        c✝ : OrderedFinpartition n
        c : OrderedFinpartition (HAdd.hAdd n 1)
        hc : Ne (Set.range (c.emb 0)) (Singleton.singleton 0)
        i : Fin c.length
        j : Fin (Function.update c.partSize (c.index 0) (HSub.hSub (c.partSize (c.inde …
        h : Eq i (c.index 0)
        ⊢ Fin n
      -/
    · refine Fin.pred (c.emb i (Fin.cast ?_ (succ j))) ?_
        /-
          case pos.refine_1
          𝕜 : Type u_1
          inst✝⁶ : NontriviallyNormedField 𝕜
          E : Type u_2
          inst✝⁵ : NormedAddCommGroup E
          inst✝⁴ : NormedSpace 𝕜 E
          F : Type u_3
          inst✝³ : NormedAddCommGroup F
          inst✝² : NormedSpace 𝕜 F
          G : Type u_4
          inst✝¹ : NormedAddCommGroup G
          inst✝ : NormedSpace 𝕜 G
          s : Set E
          t : Set F
          q : F → FormalMultilinearSeries 𝕜 F G
          p : E → FormalMultilinearSeries 𝕜 E F
          n : Nat
          c✝ : OrderedFinpartition n
          c : OrderedFinpartition (HAdd.hAdd n 1)
          hc : Ne (Set.range (c.emb 0)) (Singleton.singleton 0)
          i : Fin c.length
          j : Fin (Function.update c.partSize (c.index 0) (HSub.hSub (c.partSize (c.inde …
          h : Eq i (c.index 0)
          ⊢ Eq (HAdd.hAdd (Function.update c.partSize (c.index 0) (HSub.hSub (c.partSize …
        -/
      · rw [h]
        /-
          case pos.refine_1
          𝕜 : Type u_1
          inst✝⁶ : NontriviallyNormedField 𝕜
          E : Type u_2
          inst✝⁵ : NormedAddCommGroup E
          inst✝⁴ : NormedSpace 𝕜 E
          F : Type u_3
          inst✝³ : NormedAddCommGroup F
          inst✝² : NormedSpace 𝕜 F
          G : Type u_4
          inst✝¹ : NormedAddCommGroup G
          inst✝ : NormedSpace 𝕜 G
          s : Set E
          t : Set F
          q : F → FormalMultilinearSeries 𝕜 F G
          p : E → FormalMultilinearSeries 𝕜 E F
          n : Nat
          c✝ : OrderedFinpartition n
          c : OrderedFinpartition (HAdd.hAdd n 1)
          hc : Ne (Set.range (c.emb 0)) (Singleton.singleton 0)
          i : Fin c.length
          j : Fin (Function.update c.partSize (c.index 0) (HSub.hSub (c.partSize (c.inde …
          h : Eq i (c.index 0)
          ⊢ Eq (HAdd.hAdd (Function.update c.partSize (c.index 0) (HSub.hSub (c.partSize …
        -/
        simpa using Nat.sub_add_cancel (c.partSize_pos (c.index 0))
        /-
          🎉 no goals
        -/
        /-
          case pos.refine_2
          𝕜 : Type u_1
          inst✝⁶ : NontriviallyNormedField 𝕜
          E : Type u_2
          inst✝⁵ : NormedAddCommGroup E
          inst✝⁴ : NormedSpace 𝕜 E
          F : Type u_3
          inst✝³ : NormedAddCommGroup F
          inst✝² : NormedSpace 𝕜 F
          G : Type u_4
          inst✝¹ : NormedAddCommGroup G
          inst✝ : NormedSpace 𝕜 G
          s : Set E
          t : Set F
          q : F → FormalMultilinearSeries 𝕜 F G
          p : E → FormalMultilinearSeries 𝕜 E F
          n : Nat
          c✝ : OrderedFinpartition n
          c : OrderedFinpartition (HAdd.hAdd n 1)
          hc : Ne (Set.range (c.emb 0)) (Singleton.singleton 0)
          i : Fin c.length
          j : Fin (Function.update c.partSize (c.index 0) (HSub.hSub (c.partSize (c.inde …
          h : Eq i (c.index 0)
          ⊢ Ne (c.emb i (Fin.cast ⋯ j.succ)) 0
        -/
      · have : 0 ≤ c.emb i 0 := Fin.zero_le _
        /-
          case pos.refine_2
          𝕜 : Type u_1
          inst✝⁶ : NontriviallyNormedField 𝕜
          E : Type u_2
          inst✝⁵ : NormedAddCommGroup E
          inst✝⁴ : NormedSpace 𝕜 E
          F : Type u_3
          inst✝³ : NormedAddCommGroup F
          inst✝² : NormedSpace 𝕜 F
          G : Type u_4
          inst✝¹ : NormedAddCommGroup G
          inst✝ : NormedSpace 𝕜 G
          s : Set E
          t : Set F
          q : F → FormalMultilinearSeries 𝕜 F G
          p : E → FormalMultilinearSeries 𝕜 E F
          n : Nat
          c✝ : OrderedFinpartition n
          c : OrderedFinpartition (HAdd.hAdd n 1)
          hc : Ne (Set.range (c.emb 0)) (Singleton.singleton 0)
          i : Fin c.length
          j : Fin (Function.update c.partSize (c.index 0) (HSub.hSub (c.partSize (c.inde …
          h : Eq i (c.index 0)
          this : LE.le 0 (c.emb i 0)
          ⊢ Ne (c.emb i (Fin.cast ⋯ j.succ)) 0
        -/
        exact (this.trans_lt (c.emb_strictMono _ (succ_pos _))).ne'
        /-
          🎉 no goals
        -/
      /-
        case neg
        𝕜 : Type u_1
        inst✝⁶ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝⁵ : NormedAddCommGroup E
        inst✝⁴ : NormedSpace 𝕜 E
        F : Type u_3
        inst✝³ : NormedAddCommGroup F
        inst✝² : NormedSpace 𝕜 F
        G : Type u_4
        inst✝¹ : NormedAddCommGroup G
        inst✝ : NormedSpace 𝕜 G
        s : Set E
        t : Set F
        q : F → FormalMultilinearSeries 𝕜 F G
        p : E → FormalMultilinearSeries 𝕜 E F
        n : Nat
        c✝ : OrderedFinpartition n
        c : OrderedFinpartition (HAdd.hAdd n 1)
        hc : Ne (Set.range (c.emb 0)) (Singleton.singleton 0)
        i : Fin c.length
        j : Fin (Function.update c.partSize (c.index 0) (HSub.hSub (c.partSize (c.inde …
        h : Not (Eq i (c.index 0))
        ⊢ Fin n
      -/
    · refine Fin.pred (c.emb i (Fin.cast ?_ j)) ?_
        /-
          case neg.refine_1
          𝕜 : Type u_1
          inst✝⁶ : NontriviallyNormedField 𝕜
          E : Type u_2
          inst✝⁵ : NormedAddCommGroup E
          inst✝⁴ : NormedSpace 𝕜 E
          F : Type u_3
          inst✝³ : NormedAddCommGroup F
          inst✝² : NormedSpace 𝕜 F
          G : Type u_4
          inst✝¹ : NormedAddCommGroup G
          inst✝ : NormedSpace 𝕜 G
          s : Set E
          t : Set F
          q : F → FormalMultilinearSeries 𝕜 F G
          p : E → FormalMultilinearSeries 𝕜 E F
          n : Nat
          c✝ : OrderedFinpartition n
          c : OrderedFinpartition (HAdd.hAdd n 1)
          hc : Ne (Set.range (c.emb 0)) (Singleton.singleton 0)
          i : Fin c.length
          j : Fin (Function.update c.partSize (c.index 0) (HSub.hSub (c.partSize (c.inde …
          h : Not (Eq i (c.index 0))
          ⊢ Eq (Function.update c.partSize (c.index 0) (HSub.hSub (c.partSize (c.index 0 …
        -/
      · simp [h]
        /-
          🎉 no goals
        -/
        /-
          case neg.refine_2
          𝕜 : Type u_1
          inst✝⁶ : NontriviallyNormedField 𝕜
          E : Type u_2
          inst✝⁵ : NormedAddCommGroup E
          inst✝⁴ : NormedSpace 𝕜 E
          F : Type u_3
          inst✝³ : NormedAddCommGroup F
          inst✝² : NormedSpace 𝕜 F
          G : Type u_4
          inst✝¹ : NormedAddCommGroup G
          inst✝ : NormedSpace 𝕜 G
          s : Set E
          t : Set F
          q : F → FormalMultilinearSeries 𝕜 F G
          p : E → FormalMultilinearSeries 𝕜 E F
          n : Nat
          c✝ : OrderedFinpartition n
          c : OrderedFinpartition (HAdd.hAdd n 1)
          hc : Ne (Set.range (c.emb 0)) (Singleton.singleton 0)
          i : Fin c.length
          j : Fin (Function.update c.partSize (c.index 0) (HSub.hSub (c.partSize (c.inde …
          h : Not (Eq i (c.index 0))
          ⊢ Ne (c.emb i (Fin.cast ⋯ j)) 0
        -/
      · conv_rhs => rw [← c.emb_invEmbedding 0]
        /-
          case neg.refine_2
          𝕜 : Type u_1
          inst✝⁶ : NontriviallyNormedField 𝕜
          E : Type u_2
          inst✝⁵ : NormedAddCommGroup E
          inst✝⁴ : NormedSpace 𝕜 E
          F : Type u_3
          inst✝³ : NormedAddCommGroup F
          inst✝² : NormedSpace 𝕜 F
          G : Type u_4
          inst✝¹ : NormedAddCommGroup G
          inst✝ : NormedSpace 𝕜 G
          s : Set E
          t : Set F
          q : F → FormalMultilinearSeries 𝕜 F G
          p : E → FormalMultilinearSeries 𝕜 E F
          n : Nat
          c✝ : OrderedFinpartition n
          c : OrderedFinpartition (HAdd.hAdd n 1)
          hc : Ne (Set.range (c.emb 0)) (Singleton.singleton 0)
          i : Fin c.length
          j : Fin (Function.update c.partSize (c.index 0) (HSub.hSub (c.partSize (c.inde …
          h : Not (Eq i (c.index 0))
          ⊢ Ne (c.emb i (Fin.cast ⋯ j)) (c.emb (c.index 0) (c.invEmbedding 0))
        -/
        exact c.emb_ne_emb_of_ne h
        /-
          🎉 no goals
        -/
  emb_strictMono i a b hab := by
    /-
      𝕜 : Type u_1
      inst✝⁶ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace 𝕜 E
      F : Type u_3
      inst✝³ : NormedAddCommGroup F
      inst✝² : NormedSpace 𝕜 F
      G : Type u_4
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace 𝕜 G
      s : Set E
      t : Set F
      q : F → FormalMultilinearSeries 𝕜 F G
      p : E → FormalMultilinearSeries 𝕜 E F
      n : Nat
      c✝ : OrderedFinpartition n
      c : OrderedFinpartition (HAdd.hAdd n 1)
      hc : Ne (Set.range (c.emb 0)) (Singleton.singleton 0)
      i : Fin c.length
      a b : Fin (Function.update c.partSize (c.index 0) (HSub.hSub (c.partSize (c.in …
      hab : LT.lt a b
      ⊢ LT.lt ((fun i j => dite (Eq i (c.index 0)) (fun h => (c.emb i (Fin.cast ⋯ j. …
    -/
    rcases eq_or_ne i (c.index 0) with rfl | hi
      /-
        case inl
        𝕜 : Type u_1
        inst✝⁶ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝⁵ : NormedAddCommGroup E
        inst✝⁴ : NormedSpace 𝕜 E
        F : Type u_3
        inst✝³ : NormedAddCommGroup F
        inst✝² : NormedSpace 𝕜 F
        G : Type u_4
        inst✝¹ : NormedAddCommGroup G
        inst✝ : NormedSpace 𝕜 G
        s : Set E
        t : Set F
        q : F → FormalMultilinearSeries 𝕜 F G
        p : E → FormalMultilinearSeries 𝕜 E F
        n : Nat
        c✝ : OrderedFinpartition n
        c : OrderedFinpartition (HAdd.hAdd n 1)
        hc : Ne (Set.range (c.emb 0)) (Singleton.singleton 0)
        a b : Fin (Function.update c.partSize (c.index 0) (HSub.hSub (c.partSize (c.in …
        hab : LT.lt a b
        ⊢ LT.lt ((fun i j => dite (Eq i (c.index 0)) (fun h => (c.emb i (Fin.cast ⋯ j. …
      -/
    · simp only [↓reduceDIte, Nat.succ_eq_add_one, pred_lt_pred_iff]
      /-
        case inl
        𝕜 : Type u_1
        inst✝⁶ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝⁵ : NormedAddCommGroup E
        inst✝⁴ : NormedSpace 𝕜 E
        F : Type u_3
        inst✝³ : NormedAddCommGroup F
        inst✝² : NormedSpace 𝕜 F
        G : Type u_4
        inst✝¹ : NormedAddCommGroup G
        inst✝ : NormedSpace 𝕜 G
        s : Set E
        t : Set F
        q : F → FormalMultilinearSeries 𝕜 F G
        p : E → FormalMultilinearSeries 𝕜 E F
        n : Nat
        c✝ : OrderedFinpartition n
        c : OrderedFinpartition (HAdd.hAdd n 1)
        hc : Ne (Set.range (c.emb 0)) (Singleton.singleton 0)
        a b : Fin (Function.update c.partSize (c.index 0) (HSub.hSub (c.partSize (c.in …
        hab : LT.lt a b
        ⊢ LT.lt (c.emb (c.index 0) (Fin.cast ⋯ a.succ)) (c.emb (c.index 0) (Fin.cast ⋯ …
      -/
      exact (c.emb_strictMono _).comp (cast_strictMono _) (by simpa using hab)
      /-
        🎉 no goals
      -/
      /-
        case inr
        𝕜 : Type u_1
        inst✝⁶ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝⁵ : NormedAddCommGroup E
        inst✝⁴ : NormedSpace 𝕜 E
        F : Type u_3
        inst✝³ : NormedAddCommGroup F
        inst✝² : NormedSpace 𝕜 F
        G : Type u_4
        inst✝¹ : NormedAddCommGroup G
        inst✝ : NormedSpace 𝕜 G
        s : Set E
        t : Set F
        q : F → FormalMultilinearSeries 𝕜 F G
        p : E → FormalMultilinearSeries 𝕜 E F
        n : Nat
        c✝ : OrderedFinpartition n
        c : OrderedFinpartition (HAdd.hAdd n 1)
        hc : Ne (Set.range (c.emb 0)) (Singleton.singleton 0)
        i : Fin c.length
        a b : Fin (Function.update c.partSize (c.index 0) (HSub.hSub (c.partSize (c.in …
        hab : LT.lt a b
        hi : Ne i (c.index 0)
        ⊢ LT.lt ((fun i j => dite (Eq i (c.index 0)) (fun h => (c.emb i (Fin.cast ⋯ j. …
      -/
    · simp only [hi, ↓reduceDIte, pred_lt_pred_iff, Nat.succ_eq_add_one]
      /-
        case inr
        𝕜 : Type u_1
        inst✝⁶ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝⁵ : NormedAddCommGroup E
        inst✝⁴ : NormedSpace 𝕜 E
        F : Type u_3
        inst✝³ : NormedAddCommGroup F
        inst✝² : NormedSpace 𝕜 F
        G : Type u_4
        inst✝¹ : NormedAddCommGroup G
        inst✝ : NormedSpace 𝕜 G
        s : Set E
        t : Set F
        q : F → FormalMultilinearSeries 𝕜 F G
        p : E → FormalMultilinearSeries 𝕜 E F
        n : Nat
        c✝ : OrderedFinpartition n
        c : OrderedFinpartition (HAdd.hAdd n 1)
        hc : Ne (Set.range (c.emb 0)) (Singleton.singleton 0)
        i : Fin c.length
        a b : Fin (Function.update c.partSize (c.index 0) (HSub.hSub (c.partSize (c.in …
        hab : LT.lt a b
        hi : Ne i (c.index 0)
        ⊢ LT.lt (c.emb i (Fin.cast ⋯ a)) (c.emb i (Fin.cast ⋯ b))
      -/
      exact (c.emb_strictMono _).comp (cast_strictMono _) hab
      /-
        🎉 no goals
      -/
  parts_strictMono i j hij := by
    /-
      𝕜 : Type u_1
      inst✝⁶ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace 𝕜 E
      F : Type u_3
      inst✝³ : NormedAddCommGroup F
      inst✝² : NormedSpace 𝕜 F
      G : Type u_4
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace 𝕜 G
      s : Set E
      t : Set F
      q : F → FormalMultilinearSeries 𝕜 F G
      p : E → FormalMultilinearSeries 𝕜 E F
      n : Nat
      c✝ : OrderedFinpartition n
      c : OrderedFinpartition (HAdd.hAdd n 1)
      hc : Ne (Set.range (c.emb 0)) (Singleton.singleton 0)
      i j : Fin c.length
      hij : LT.lt i j
      ⊢ LT.lt ((fun m => (fun i j => dite (Eq i (c.index 0)) (fun h => (c.emb i (Fin …
    -/
    simp only [Fin.lt_iff_val_lt_val]
    /-
      𝕜 : Type u_1
      inst✝⁶ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace 𝕜 E
      F : Type u_3
      inst✝³ : NormedAddCommGroup F
      inst✝² : NormedSpace 𝕜 F
      G : Type u_4
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace 𝕜 G
      s : Set E
      t : Set F
      q : F → FormalMultilinearSeries 𝕜 F G
      p : E → FormalMultilinearSeries 𝕜 E F
      n : Nat
      c✝ : OrderedFinpartition n
      c : OrderedFinpartition (HAdd.hAdd n 1)
      hc : Ne (Set.range (c.emb 0)) (Singleton.singleton 0)
      i j : Fin c.length
      hij : LT.lt i j
      ⊢ LT.lt ↑(dite (Eq i (c.index 0)) (fun h => (c.emb i (Fin.cast ⋯ ⟨HSub.hSub (F …
    -/
    rw [← Nat.add_lt_add_iff_right (k := 1)]
    /-
      𝕜 : Type u_1
      inst✝⁶ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace 𝕜 E
      F : Type u_3
      inst✝³ : NormedAddCommGroup F
      inst✝² : NormedSpace 𝕜 F
      G : Type u_4
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace 𝕜 G
      s : Set E
      t : Set F
      q : F → FormalMultilinearSeries 𝕜 F G
      p : E → FormalMultilinearSeries 𝕜 E F
      n : Nat
      c✝ : OrderedFinpartition n
      c : OrderedFinpartition (HAdd.hAdd n 1)
      hc : Ne (Set.range (c.emb 0)) (Singleton.singleton 0)
      i j : Fin c.length
      hij : LT.lt i j
      ⊢ LT.lt (HAdd.hAdd (↑(dite (Eq i (c.index 0)) (fun h => (c.emb i (Fin.cast ⋯ ⟨ …
    -/
    convert Fin.lt_iff_val_lt_val.1 (c.parts_strictMono hij)
      /-
        case h.e'_3
        𝕜 : Type u_1
        inst✝⁶ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝⁵ : NormedAddCommGroup E
        inst✝⁴ : NormedSpace 𝕜 E
        F : Type u_3
        inst✝³ : NormedAddCommGroup F
        inst✝² : NormedSpace 𝕜 F
        G : Type u_4
        inst✝¹ : NormedAddCommGroup G
        inst✝ : NormedSpace 𝕜 G
        s : Set E
        t : Set F
        q : F → FormalMultilinearSeries 𝕜 F G
        p : E → FormalMultilinearSeries 𝕜 E F
        n : Nat
        c✝ : OrderedFinpartition n
        c : OrderedFinpartition (HAdd.hAdd n 1)
        hc : Ne (Set.range (c.emb 0)) (Singleton.singleton 0)
        i j : Fin c.length
        hij : LT.lt i j
        ⊢ Eq (HAdd.hAdd (↑(dite (Eq i (c.index 0)) (fun h => (c.emb i (Fin.cast ⋯ ⟨HSu …
      -/
    · rcases eq_or_ne i (c.index 0) with rfl | hi
        /-
          case h.e'_3.inl
          𝕜 : Type u_1
          inst✝⁶ : NontriviallyNormedField 𝕜
          E : Type u_2
          inst✝⁵ : NormedAddCommGroup E
          inst✝⁴ : NormedSpace 𝕜 E
          F : Type u_3
          inst✝³ : NormedAddCommGroup F
          inst✝² : NormedSpace 𝕜 F
          G : Type u_4
          inst✝¹ : NormedAddCommGroup G
          inst✝ : NormedSpace 𝕜 G
          s : Set E
          t : Set F
          q : F → FormalMultilinearSeries 𝕜 F G
          p : E → FormalMultilinearSeries 𝕜 E F
          n : Nat
          c✝ : OrderedFinpartition n
          c : OrderedFinpartition (HAdd.hAdd n 1)
          hc : Ne (Set.range (c.emb 0)) (Singleton.singleton 0)
          j : Fin c.length
          hij : LT.lt (c.index 0) j
          ⊢ Eq (HAdd.hAdd (↑(dite (Eq (c.index 0) (c.index 0)) (fun h => (c.emb (c.index …
        -/
      · simp only [↓reduceDIte, Nat.succ_eq_add_one, update_self, succ_mk, cast_mk, coe_pred]
        /-
          case h.e'_3.inl
          𝕜 : Type u_1
          inst✝⁶ : NontriviallyNormedField 𝕜
          E : Type u_2
          inst✝⁵ : NormedAddCommGroup E
          inst✝⁴ : NormedSpace 𝕜 E
          F : Type u_3
          inst✝³ : NormedAddCommGroup F
          inst✝² : NormedSpace 𝕜 F
          G : Type u_4
          inst✝¹ : NormedAddCommGroup G
          inst✝ : NormedSpace 𝕜 G
          s : Set E
          t : Set F
          q : F → FormalMultilinearSeries 𝕜 F G
          p : E → FormalMultilinearSeries 𝕜 E F
          n : Nat
          c✝ : OrderedFinpartition n
          c : OrderedFinpartition (HAdd.hAdd n 1)
          hc : Ne (Set.range (c.emb 0)) (Singleton.singleton 0)
          j : Fin c.length
          hij : LT.lt (c.index 0) j
          ⊢ Eq (HAdd.hAdd (HSub.hSub (↑(c.emb (c.index 0) ⟨HAdd.hAdd (HSub.hSub (HSub.hS …
        -/
        have A := c.one_lt_partSize_index_zero hc
        /-
          case h.e'_3.inl
          𝕜 : Type u_1
          inst✝⁶ : NontriviallyNormedField 𝕜
          E : Type u_2
          inst✝⁵ : NormedAddCommGroup E
          inst✝⁴ : NormedSpace 𝕜 E
          F : Type u_3
          inst✝³ : NormedAddCommGroup F
          inst✝² : NormedSpace 𝕜 F
          G : Type u_4
          inst✝¹ : NormedAddCommGroup G
          inst✝ : NormedSpace 𝕜 G
          s : Set E
          t : Set F
          q : F → FormalMultilinearSeries 𝕜 F G
          p : E → FormalMultilinearSeries 𝕜 E F
          n : Nat
          c✝ : OrderedFinpartition n
          c : OrderedFinpartition (HAdd.hAdd n 1)
          hc : Ne (Set.range (c.emb 0)) (Singleton.singleton 0)
          j : Fin c.length
          hij : LT.lt (c.index 0) j
          A : LT.lt 1 (c.partSize (c.index 0))
          ⊢ Eq (HAdd.hAdd (HSub.hSub (↑(c.emb (c.index 0) ⟨HAdd.hAdd (HSub.hSub (HSub.hS …
        -/
        rw [Nat.sub_add_cancel]
          /-
            case h.e'_3.inl
            𝕜 : Type u_1
            inst✝⁶ : NontriviallyNormedField 𝕜
            E : Type u_2
            inst✝⁵ : NormedAddCommGroup E
            inst✝⁴ : NormedSpace 𝕜 E
            F : Type u_3
            inst✝³ : NormedAddCommGroup F
            inst✝² : NormedSpace 𝕜 F
            G : Type u_4
            inst✝¹ : NormedAddCommGroup G
            inst✝ : NormedSpace 𝕜 G
            s : Set E
            t : Set F
            q : F → FormalMultilinearSeries 𝕜 F G
            p : E → FormalMultilinearSeries 𝕜 E F
            n : Nat
            c✝ : OrderedFinpartition n
            c : OrderedFinpartition (HAdd.hAdd n 1)
            hc : Ne (Set.range (c.emb 0)) (Singleton.singleton 0)
            j : Fin c.length
            hij : LT.lt (c.index 0) j
            A : LT.lt 1 (c.partSize (c.index 0))
            ⊢ Eq ↑(c.emb (c.index 0) ⟨HAdd.hAdd (HSub.hSub (HSub.hSub (c.partSize (c.index …
          -/
        · congr; omega
                 /-
                   🎉 no goals
                 -/
          /-
            case h.e'_3.inl
            𝕜 : Type u_1
            inst✝⁶ : NontriviallyNormedField 𝕜
            E : Type u_2
            inst✝⁵ : NormedAddCommGroup E
            inst✝⁴ : NormedSpace 𝕜 E
            F : Type u_3
            inst✝³ : NormedAddCommGroup F
            inst✝² : NormedSpace 𝕜 F
            G : Type u_4
            inst✝¹ : NormedAddCommGroup G
            inst✝ : NormedSpace 𝕜 G
            s : Set E
            t : Set F
            q : F → FormalMultilinearSeries 𝕜 F G
            p : E → FormalMultilinearSeries 𝕜 E F
            n : Nat
            c✝ : OrderedFinpartition n
            c : OrderedFinpartition (HAdd.hAdd n 1)
            hc : Ne (Set.range (c.emb 0)) (Singleton.singleton 0)
            j : Fin c.length
            hij : LT.lt (c.index 0) j
            A : LT.lt 1 (c.partSize (c.index 0))
            ⊢ LE.le 1 ↑(c.emb (c.index 0) ⟨HAdd.hAdd (HSub.hSub (HSub.hSub (c.partSize (c. …
          -/
        · rw [Order.one_le_iff_pos]
          /-
            case h.e'_3.inl
            𝕜 : Type u_1
            inst✝⁶ : NontriviallyNormedField 𝕜
            E : Type u_2
            inst✝⁵ : NormedAddCommGroup E
            inst✝⁴ : NormedSpace 𝕜 E
            F : Type u_3
            inst✝³ : NormedAddCommGroup F
            inst✝² : NormedSpace 𝕜 F
            G : Type u_4
            inst✝¹ : NormedAddCommGroup G
            inst✝ : NormedSpace 𝕜 G
            s : Set E
            t : Set F
            q : F → FormalMultilinearSeries 𝕜 F G
            p : E → FormalMultilinearSeries 𝕜 E F
            n : Nat
            c✝ : OrderedFinpartition n
            c : OrderedFinpartition (HAdd.hAdd n 1)
            hc : Ne (Set.range (c.emb 0)) (Singleton.singleton 0)
            j : Fin c.length
            hij : LT.lt (c.index 0) j
            A : LT.lt 1 (c.partSize (c.index 0))
            ⊢ LT.lt 0 ↑(c.emb (c.index 0) ⟨HAdd.hAdd (HSub.hSub (HSub.hSub (c.partSize (c. …
          -/
          conv_lhs => rw [show (0 : ℕ) = c.emb (c.index 0) 0 by simp [emb_zero]]
          /-
            case h.e'_3.inl
            𝕜 : Type u_1
            inst✝⁶ : NontriviallyNormedField 𝕜
            E : Type u_2
            inst✝⁵ : NormedAddCommGroup E
            inst✝⁴ : NormedSpace 𝕜 E
            F : Type u_3
            inst✝³ : NormedAddCommGroup F
            inst✝² : NormedSpace 𝕜 F
            G : Type u_4
            inst✝¹ : NormedAddCommGroup G
            inst✝ : NormedSpace 𝕜 G
            s : Set E
            t : Set F
            q : F → FormalMultilinearSeries 𝕜 F G
            p : E → FormalMultilinearSeries 𝕜 E F
            n : Nat
            c✝ : OrderedFinpartition n
            c : OrderedFinpartition (HAdd.hAdd n 1)
            hc : Ne (Set.range (c.emb 0)) (Singleton.singleton 0)
            j : Fin c.length
            hij : LT.lt (c.index 0) j
            A : LT.lt 1 (c.partSize (c.index 0))
            ⊢ LT.lt ↑(c.emb (c.index 0) 0) ↑(c.emb (c.index 0) ⟨HAdd.hAdd (HSub.hSub (HSub …
          -/
          rw [← lt_iff_val_lt_val]
          /-
            case h.e'_3.inl
            𝕜 : Type u_1
            inst✝⁶ : NontriviallyNormedField 𝕜
            E : Type u_2
            inst✝⁵ : NormedAddCommGroup E
            inst✝⁴ : NormedSpace 𝕜 E
            F : Type u_3
            inst✝³ : NormedAddCommGroup F
            inst✝² : NormedSpace 𝕜 F
            G : Type u_4
            inst✝¹ : NormedAddCommGroup G
            inst✝ : NormedSpace 𝕜 G
            s : Set E
            t : Set F
            q : F → FormalMultilinearSeries 𝕜 F G
            p : E → FormalMultilinearSeries 𝕜 E F
            n : Nat
            c✝ : OrderedFinpartition n
            c : OrderedFinpartition (HAdd.hAdd n 1)
            hc : Ne (Set.range (c.emb 0)) (Singleton.singleton 0)
            j : Fin c.length
            hij : LT.lt (c.index 0) j
            A : LT.lt 1 (c.partSize (c.index 0))
            ⊢ LT.lt (c.emb (c.index 0) 0) (c.emb (c.index 0) ⟨HAdd.hAdd (HSub.hSub (HSub.h …
          -/
          apply c.emb_strictMono
          /-
            case h.e'_3.inl.a
            𝕜 : Type u_1
            inst✝⁶ : NontriviallyNormedField 𝕜
            E : Type u_2
            inst✝⁵ : NormedAddCommGroup E
            inst✝⁴ : NormedSpace 𝕜 E
            F : Type u_3
            inst✝³ : NormedAddCommGroup F
            inst✝² : NormedSpace 𝕜 F
            G : Type u_4
            inst✝¹ : NormedAddCommGroup G
            inst✝ : NormedSpace 𝕜 G
            s : Set E
            t : Set F
            q : F → FormalMultilinearSeries 𝕜 F G
            p : E → FormalMultilinearSeries 𝕜 E F
            n : Nat
            c✝ : OrderedFinpartition n
            c : OrderedFinpartition (HAdd.hAdd n 1)
            hc : Ne (Set.range (c.emb 0)) (Singleton.singleton 0)
            j : Fin c.length
            hij : LT.lt (c.index 0) j
            A : LT.lt 1 (c.partSize (c.index 0))
            ⊢ LT.lt 0 ⟨HAdd.hAdd (HSub.hSub (HSub.hSub (c.partSize (c.index 0)) 1) 1) 1, ⋯⟩
          -/
          simp [lt_iff_val_lt_val]
          /-
            🎉 no goals
          -/
        /-
          case h.e'_3.inr
          𝕜 : Type u_1
          inst✝⁶ : NontriviallyNormedField 𝕜
          E : Type u_2
          inst✝⁵ : NormedAddCommGroup E
          inst✝⁴ : NormedSpace 𝕜 E
          F : Type u_3
          inst✝³ : NormedAddCommGroup F
          inst✝² : NormedSpace 𝕜 F
          G : Type u_4
          inst✝¹ : NormedAddCommGroup G
          inst✝ : NormedSpace 𝕜 G
          s : Set E
          t : Set F
          q : F → FormalMultilinearSeries 𝕜 F G
          p : E → FormalMultilinearSeries 𝕜 E F
          n : Nat
          c✝ : OrderedFinpartition n
          c : OrderedFinpartition (HAdd.hAdd n 1)
          hc : Ne (Set.range (c.emb 0)) (Singleton.singleton 0)
          i j : Fin c.length
          hij : LT.lt i j
          hi : Ne i (c.index 0)
          ⊢ Eq (HAdd.hAdd (↑(dite (Eq i (c.index 0)) (fun h => (c.emb i (Fin.cast ⋯ ⟨HSu …
        -/
      · simp only [hi, ↓reduceDIte, ne_eq, not_false_eq_true, update_of_ne, cast_mk, coe_pred]
        /-
          case h.e'_3.inr
          𝕜 : Type u_1
          inst✝⁶ : NontriviallyNormedField 𝕜
          E : Type u_2
          inst✝⁵ : NormedAddCommGroup E
          inst✝⁴ : NormedSpace 𝕜 E
          F : Type u_3
          inst✝³ : NormedAddCommGroup F
          inst✝² : NormedSpace 𝕜 F
          G : Type u_4
          inst✝¹ : NormedAddCommGroup G
          inst✝ : NormedSpace 𝕜 G
          s : Set E
          t : Set F
          q : F → FormalMultilinearSeries 𝕜 F G
          p : E → FormalMultilinearSeries 𝕜 E F
          n : Nat
          c✝ : OrderedFinpartition n
          c : OrderedFinpartition (HAdd.hAdd n 1)
          hc : Ne (Set.range (c.emb 0)) (Singleton.singleton 0)
          i j : Fin c.length
          hij : LT.lt i j
          hi : Ne i (c.index 0)
          ⊢ Eq (HAdd.hAdd (HSub.hSub (↑(c.emb i ⟨HSub.hSub (c.partSize i) 1, ⋯⟩)) 1) 1)  …
        -/
        apply Nat.sub_add_cancel
        have : c.emb i ⟨c.partSize i - 1, Nat.sub_one_lt_of_lt (c.partSize_pos i)⟩
            ≠ c.emb (c.index 0) 0 := c.emb_ne_emb_of_ne hi
        /-
          case h.e'_3.inr.h
          𝕜 : Type u_1
          inst✝⁶ : NontriviallyNormedField 𝕜
          E : Type u_2
          inst✝⁵ : NormedAddCommGroup E
          inst✝⁴ : NormedSpace 𝕜 E
          F : Type u_3
          inst✝³ : NormedAddCommGroup F
          inst✝² : NormedSpace 𝕜 F
          G : Type u_4
          inst✝¹ : NormedAddCommGroup G
          inst✝ : NormedSpace 𝕜 G
          s : Set E
          t : Set F
          q : F → FormalMultilinearSeries 𝕜 F G
          p : E → FormalMultilinearSeries 𝕜 E F
          n : Nat
          c✝ : OrderedFinpartition n
          c : OrderedFinpartition (HAdd.hAdd n 1)
          hc : Ne (Set.range (c.emb 0)) (Singleton.singleton 0)
          i j : Fin c.length
          hij : LT.lt i j
          hi : Ne i (c.index 0)
          this : Ne (c.emb i ⟨HSub.hSub (c.partSize i) 1, ⋯⟩) (c.emb (c.index 0) 0)
          ⊢ LE.le 1 ↑(c.emb i ⟨HSub.hSub (c.partSize i) 1, ⋯⟩)
        -/
        simp only [c.emb_zero, ne_eq, ← val_eq_val, val_zero] at this
        /-
          case h.e'_3.inr.h
          𝕜 : Type u_1
          inst✝⁶ : NontriviallyNormedField 𝕜
          E : Type u_2
          inst✝⁵ : NormedAddCommGroup E
          inst✝⁴ : NormedSpace 𝕜 E
          F : Type u_3
          inst✝³ : NormedAddCommGroup F
          inst✝² : NormedSpace 𝕜 F
          G : Type u_4
          inst✝¹ : NormedAddCommGroup G
          inst✝ : NormedSpace 𝕜 G
          s : Set E
          t : Set F
          q : F → FormalMultilinearSeries 𝕜 F G
          p : E → FormalMultilinearSeries 𝕜 E F
          n : Nat
          c✝ : OrderedFinpartition n
          c : OrderedFinpartition (HAdd.hAdd n 1)
          hc : Ne (Set.range (c.emb 0)) (Singleton.singleton 0)
          i j : Fin c.length
          hij : LT.lt i j
          hi : Ne i (c.index 0)
          this : Not (Eq (↑(c.emb i ⟨HSub.hSub (c.partSize i) 1, ⋯⟩)) 0)
          ⊢ LE.le 1 ↑(c.emb i ⟨HSub.hSub (c.partSize i) 1, ⋯⟩)
        -/
        omega
        /-
          🎉 no goals
        -/
      /-
        case h.e'_4
        𝕜 : Type u_1
        inst✝⁶ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝⁵ : NormedAddCommGroup E
        inst✝⁴ : NormedSpace 𝕜 E
        F : Type u_3
        inst✝³ : NormedAddCommGroup F
        inst✝² : NormedSpace 𝕜 F
        G : Type u_4
        inst✝¹ : NormedAddCommGroup G
        inst✝ : NormedSpace 𝕜 G
        s : Set E
        t : Set F
        q : F → FormalMultilinearSeries 𝕜 F G
        p : E → FormalMultilinearSeries 𝕜 E F
        n : Nat
        c✝ : OrderedFinpartition n
        c : OrderedFinpartition (HAdd.hAdd n 1)
        hc : Ne (Set.range (c.emb 0)) (Singleton.singleton 0)
        i j : Fin c.length
        hij : LT.lt i j
        ⊢ Eq (HAdd.hAdd (↑(dite (Eq j (c.index 0)) (fun h => (c.emb j (Fin.cast ⋯ ⟨HSu …
      -/
    · rcases eq_or_ne j (c.index 0) with rfl | hj
        /-
          case h.e'_4.inl
          𝕜 : Type u_1
          inst✝⁶ : NontriviallyNormedField 𝕜
          E : Type u_2
          inst✝⁵ : NormedAddCommGroup E
          inst✝⁴ : NormedSpace 𝕜 E
          F : Type u_3
          inst✝³ : NormedAddCommGroup F
          inst✝² : NormedSpace 𝕜 F
          G : Type u_4
          inst✝¹ : NormedAddCommGroup G
          inst✝ : NormedSpace 𝕜 G
          s : Set E
          t : Set F
          q : F → FormalMultilinearSeries 𝕜 F G
          p : E → FormalMultilinearSeries 𝕜 E F
          n : Nat
          c✝ : OrderedFinpartition n
          c : OrderedFinpartition (HAdd.hAdd n 1)
          hc : Ne (Set.range (c.emb 0)) (Singleton.singleton 0)
          i : Fin c.length
          hij : LT.lt i (c.index 0)
          ⊢ Eq (HAdd.hAdd (↑(dite (Eq (c.index 0) (c.index 0)) (fun h => (c.emb (c.index …
        -/
      · simp only [↓reduceDIte, Nat.succ_eq_add_one, update_self, succ_mk, cast_mk, coe_pred]
        /-
          case h.e'_4.inl
          𝕜 : Type u_1
          inst✝⁶ : NontriviallyNormedField 𝕜
          E : Type u_2
          inst✝⁵ : NormedAddCommGroup E
          inst✝⁴ : NormedSpace 𝕜 E
          F : Type u_3
          inst✝³ : NormedAddCommGroup F
          inst✝² : NormedSpace 𝕜 F
          G : Type u_4
          inst✝¹ : NormedAddCommGroup G
          inst✝ : NormedSpace 𝕜 G
          s : Set E
          t : Set F
          q : F → FormalMultilinearSeries 𝕜 F G
          p : E → FormalMultilinearSeries 𝕜 E F
          n : Nat
          c✝ : OrderedFinpartition n
          c : OrderedFinpartition (HAdd.hAdd n 1)
          hc : Ne (Set.range (c.emb 0)) (Singleton.singleton 0)
          i : Fin c.length
          hij : LT.lt i (c.index 0)
          ⊢ Eq (HAdd.hAdd (HSub.hSub (↑(c.emb (c.index 0) ⟨HAdd.hAdd (HSub.hSub (HSub.hS …
        -/
        have A := c.one_lt_partSize_index_zero hc
        /-
          case h.e'_4.inl
          𝕜 : Type u_1
          inst✝⁶ : NontriviallyNormedField 𝕜
          E : Type u_2
          inst✝⁵ : NormedAddCommGroup E
          inst✝⁴ : NormedSpace 𝕜 E
          F : Type u_3
          inst✝³ : NormedAddCommGroup F
          inst✝² : NormedSpace 𝕜 F
          G : Type u_4
          inst✝¹ : NormedAddCommGroup G
          inst✝ : NormedSpace 𝕜 G
          s : Set E
          t : Set F
          q : F → FormalMultilinearSeries 𝕜 F G
          p : E → FormalMultilinearSeries 𝕜 E F
          n : Nat
          c✝ : OrderedFinpartition n
          c : OrderedFinpartition (HAdd.hAdd n 1)
          hc : Ne (Set.range (c.emb 0)) (Singleton.singleton 0)
          i : Fin c.length
          hij : LT.lt i (c.index 0)
          A : LT.lt 1 (c.partSize (c.index 0))
          ⊢ Eq (HAdd.hAdd (HSub.hSub (↑(c.emb (c.index 0) ⟨HAdd.hAdd (HSub.hSub (HSub.hS …
        -/
        rw [Nat.sub_add_cancel]
          /-
            case h.e'_4.inl
            𝕜 : Type u_1
            inst✝⁶ : NontriviallyNormedField 𝕜
            E : Type u_2
            inst✝⁵ : NormedAddCommGroup E
            inst✝⁴ : NormedSpace 𝕜 E
            F : Type u_3
            inst✝³ : NormedAddCommGroup F
            inst✝² : NormedSpace 𝕜 F
            G : Type u_4
            inst✝¹ : NormedAddCommGroup G
            inst✝ : NormedSpace 𝕜 G
            s : Set E
            t : Set F
            q : F → FormalMultilinearSeries 𝕜 F G
            p : E → FormalMultilinearSeries 𝕜 E F
            n : Nat
            c✝ : OrderedFinpartition n
            c : OrderedFinpartition (HAdd.hAdd n 1)
            hc : Ne (Set.range (c.emb 0)) (Singleton.singleton 0)
            i : Fin c.length
            hij : LT.lt i (c.index 0)
            A : LT.lt 1 (c.partSize (c.index 0))
            ⊢ Eq ↑(c.emb (c.index 0) ⟨HAdd.hAdd (HSub.hSub (HSub.hSub (c.partSize (c.index …
          -/
        · congr; omega
                 /-
                   🎉 no goals
                 -/
          /-
            case h.e'_4.inl
            𝕜 : Type u_1
            inst✝⁶ : NontriviallyNormedField 𝕜
            E : Type u_2
            inst✝⁵ : NormedAddCommGroup E
            inst✝⁴ : NormedSpace 𝕜 E
            F : Type u_3
            inst✝³ : NormedAddCommGroup F
            inst✝² : NormedSpace 𝕜 F
            G : Type u_4
            inst✝¹ : NormedAddCommGroup G
            inst✝ : NormedSpace 𝕜 G
            s : Set E
            t : Set F
            q : F → FormalMultilinearSeries 𝕜 F G
            p : E → FormalMultilinearSeries 𝕜 E F
            n : Nat
            c✝ : OrderedFinpartition n
            c : OrderedFinpartition (HAdd.hAdd n 1)
            hc : Ne (Set.range (c.emb 0)) (Singleton.singleton 0)
            i : Fin c.length
            hij : LT.lt i (c.index 0)
            A : LT.lt 1 (c.partSize (c.index 0))
            ⊢ LE.le 1 ↑(c.emb (c.index 0) ⟨HAdd.hAdd (HSub.hSub (HSub.hSub (c.partSize (c. …
          -/
        · rw [Order.one_le_iff_pos]
          /-
            case h.e'_4.inl
            𝕜 : Type u_1
            inst✝⁶ : NontriviallyNormedField 𝕜
            E : Type u_2
            inst✝⁵ : NormedAddCommGroup E
            inst✝⁴ : NormedSpace 𝕜 E
            F : Type u_3
            inst✝³ : NormedAddCommGroup F
            inst✝² : NormedSpace 𝕜 F
            G : Type u_4
            inst✝¹ : NormedAddCommGroup G
            inst✝ : NormedSpace 𝕜 G
            s : Set E
            t : Set F
            q : F → FormalMultilinearSeries 𝕜 F G
            p : E → FormalMultilinearSeries 𝕜 E F
            n : Nat
            c✝ : OrderedFinpartition n
            c : OrderedFinpartition (HAdd.hAdd n 1)
            hc : Ne (Set.range (c.emb 0)) (Singleton.singleton 0)
            i : Fin c.length
            hij : LT.lt i (c.index 0)
            A : LT.lt 1 (c.partSize (c.index 0))
            ⊢ LT.lt 0 ↑(c.emb (c.index 0) ⟨HAdd.hAdd (HSub.hSub (HSub.hSub (c.partSize (c. …
          -/
          conv_lhs => rw [show (0 : ℕ) = c.emb (c.index 0) 0 by simp [emb_zero]]
          /-
            case h.e'_4.inl
            𝕜 : Type u_1
            inst✝⁶ : NontriviallyNormedField 𝕜
            E : Type u_2
            inst✝⁵ : NormedAddCommGroup E
            inst✝⁴ : NormedSpace 𝕜 E
            F : Type u_3
            inst✝³ : NormedAddCommGroup F
            inst✝² : NormedSpace 𝕜 F
            G : Type u_4
            inst✝¹ : NormedAddCommGroup G
            inst✝ : NormedSpace 𝕜 G
            s : Set E
            t : Set F
            q : F → FormalMultilinearSeries 𝕜 F G
            p : E → FormalMultilinearSeries 𝕜 E F
            n : Nat
            c✝ : OrderedFinpartition n
            c : OrderedFinpartition (HAdd.hAdd n 1)
            hc : Ne (Set.range (c.emb 0)) (Singleton.singleton 0)
            i : Fin c.length
            hij : LT.lt i (c.index 0)
            A : LT.lt 1 (c.partSize (c.index 0))
            ⊢ LT.lt ↑(c.emb (c.index 0) 0) ↑(c.emb (c.index 0) ⟨HAdd.hAdd (HSub.hSub (HSub …
          -/
          rw [← lt_iff_val_lt_val]
          /-
            case h.e'_4.inl
            𝕜 : Type u_1
            inst✝⁶ : NontriviallyNormedField 𝕜
            E : Type u_2
            inst✝⁵ : NormedAddCommGroup E
            inst✝⁴ : NormedSpace 𝕜 E
            F : Type u_3
            inst✝³ : NormedAddCommGroup F
            inst✝² : NormedSpace 𝕜 F
            G : Type u_4
            inst✝¹ : NormedAddCommGroup G
            inst✝ : NormedSpace 𝕜 G
            s : Set E
            t : Set F
            q : F → FormalMultilinearSeries 𝕜 F G
            p : E → FormalMultilinearSeries 𝕜 E F
            n : Nat
            c✝ : OrderedFinpartition n
            c : OrderedFinpartition (HAdd.hAdd n 1)
            hc : Ne (Set.range (c.emb 0)) (Singleton.singleton 0)
            i : Fin c.length
            hij : LT.lt i (c.index 0)
            A : LT.lt 1 (c.partSize (c.index 0))
            ⊢ LT.lt (c.emb (c.index 0) 0) (c.emb (c.index 0) ⟨HAdd.hAdd (HSub.hSub (HSub.h …
          -/
          apply c.emb_strictMono
          /-
            case h.e'_4.inl.a
            𝕜 : Type u_1
            inst✝⁶ : NontriviallyNormedField 𝕜
            E : Type u_2
            inst✝⁵ : NormedAddCommGroup E
            inst✝⁴ : NormedSpace 𝕜 E
            F : Type u_3
            inst✝³ : NormedAddCommGroup F
            inst✝² : NormedSpace 𝕜 F
            G : Type u_4
            inst✝¹ : NormedAddCommGroup G
            inst✝ : NormedSpace 𝕜 G
            s : Set E
            t : Set F
            q : F → FormalMultilinearSeries 𝕜 F G
            p : E → FormalMultilinearSeries 𝕜 E F
            n : Nat
            c✝ : OrderedFinpartition n
            c : OrderedFinpartition (HAdd.hAdd n 1)
            hc : Ne (Set.range (c.emb 0)) (Singleton.singleton 0)
            i : Fin c.length
            hij : LT.lt i (c.index 0)
            A : LT.lt 1 (c.partSize (c.index 0))
            ⊢ LT.lt 0 ⟨HAdd.hAdd (HSub.hSub (HSub.hSub (c.partSize (c.index 0)) 1) 1) 1, ⋯⟩
          -/
          simp [lt_iff_val_lt_val]
          /-
            🎉 no goals
          -/
        /-
          case h.e'_4.inr
          𝕜 : Type u_1
          inst✝⁶ : NontriviallyNormedField 𝕜
          E : Type u_2
          inst✝⁵ : NormedAddCommGroup E
          inst✝⁴ : NormedSpace 𝕜 E
          F : Type u_3
          inst✝³ : NormedAddCommGroup F
          inst✝² : NormedSpace 𝕜 F
          G : Type u_4
          inst✝¹ : NormedAddCommGroup G
          inst✝ : NormedSpace 𝕜 G
          s : Set E
          t : Set F
          q : F → FormalMultilinearSeries 𝕜 F G
          p : E → FormalMultilinearSeries 𝕜 E F
          n : Nat
          c✝ : OrderedFinpartition n
          c : OrderedFinpartition (HAdd.hAdd n 1)
          hc : Ne (Set.range (c.emb 0)) (Singleton.singleton 0)
          i j : Fin c.length
          hij : LT.lt i j
          hj : Ne j (c.index 0)
          ⊢ Eq (HAdd.hAdd (↑(dite (Eq j (c.index 0)) (fun h => (c.emb j (Fin.cast ⋯ ⟨HSu …
        -/
      · simp only [hj, ↓reduceDIte, ne_eq, not_false_eq_true, update_of_ne, cast_mk, coe_pred]
        /-
          case h.e'_4.inr
          𝕜 : Type u_1
          inst✝⁶ : NontriviallyNormedField 𝕜
          E : Type u_2
          inst✝⁵ : NormedAddCommGroup E
          inst✝⁴ : NormedSpace 𝕜 E
          F : Type u_3
          inst✝³ : NormedAddCommGroup F
          inst✝² : NormedSpace 𝕜 F
          G : Type u_4
          inst✝¹ : NormedAddCommGroup G
          inst✝ : NormedSpace 𝕜 G
          s : Set E
          t : Set F
          q : F → FormalMultilinearSeries 𝕜 F G
          p : E → FormalMultilinearSeries 𝕜 E F
          n : Nat
          c✝ : OrderedFinpartition n
          c : OrderedFinpartition (HAdd.hAdd n 1)
          hc : Ne (Set.range (c.emb 0)) (Singleton.singleton 0)
          i j : Fin c.length
          hij : LT.lt i j
          hj : Ne j (c.index 0)
          ⊢ Eq (HAdd.hAdd (HSub.hSub (↑(c.emb j ⟨HSub.hSub (c.partSize j) 1, ⋯⟩)) 1) 1)  …
        -/
        apply Nat.sub_add_cancel
        have : c.emb j ⟨c.partSize j - 1, Nat.sub_one_lt_of_lt (c.partSize_pos j)⟩
            ≠ c.emb (c.index 0) 0 := c.emb_ne_emb_of_ne hj
        /-
          case h.e'_4.inr.h
          𝕜 : Type u_1
          inst✝⁶ : NontriviallyNormedField 𝕜
          E : Type u_2
          inst✝⁵ : NormedAddCommGroup E
          inst✝⁴ : NormedSpace 𝕜 E
          F : Type u_3
          inst✝³ : NormedAddCommGroup F
          inst✝² : NormedSpace 𝕜 F
          G : Type u_4
          inst✝¹ : NormedAddCommGroup G
          inst✝ : NormedSpace 𝕜 G
          s : Set E
          t : Set F
          q : F → FormalMultilinearSeries 𝕜 F G
          p : E → FormalMultilinearSeries 𝕜 E F
          n : Nat
          c✝ : OrderedFinpartition n
          c : OrderedFinpartition (HAdd.hAdd n 1)
          hc : Ne (Set.range (c.emb 0)) (Singleton.singleton 0)
          i j : Fin c.length
          hij : LT.lt i j
          hj : Ne j (c.index 0)
          this : Ne (c.emb j ⟨HSub.hSub (c.partSize j) 1, ⋯⟩) (c.emb (c.index 0) 0)
          ⊢ LE.le 1 ↑(c.emb j ⟨HSub.hSub (c.partSize j) 1, ⋯⟩)
        -/
        simp only [c.emb_zero, ne_eq, ← val_eq_val, val_zero] at this
        /-
          case h.e'_4.inr.h
          𝕜 : Type u_1
          inst✝⁶ : NontriviallyNormedField 𝕜
          E : Type u_2
          inst✝⁵ : NormedAddCommGroup E
          inst✝⁴ : NormedSpace 𝕜 E
          F : Type u_3
          inst✝³ : NormedAddCommGroup F
          inst✝² : NormedSpace 𝕜 F
          G : Type u_4
          inst✝¹ : NormedAddCommGroup G
          inst✝ : NormedSpace 𝕜 G
          s : Set E
          t : Set F
          q : F → FormalMultilinearSeries 𝕜 F G
          p : E → FormalMultilinearSeries 𝕜 E F
          n : Nat
          c✝ : OrderedFinpartition n
          c : OrderedFinpartition (HAdd.hAdd n 1)
          hc : Ne (Set.range (c.emb 0)) (Singleton.singleton 0)
          i j : Fin c.length
          hij : LT.lt i j
          hj : Ne j (c.index 0)
          this : Not (Eq (↑(c.emb j ⟨HSub.hSub (c.partSize j) 1, ⋯⟩)) 0)
          ⊢ LE.le 1 ↑(c.emb j ⟨HSub.hSub (c.partSize j) 1, ⋯⟩)
        -/
        omega
        /-
          🎉 no goals
        -/
  disjoint i _ j _ hij := by
    /-
      𝕜 : Type u_1
      inst✝⁶ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace 𝕜 E
      F : Type u_3
      inst✝³ : NormedAddCommGroup F
      inst✝² : NormedSpace 𝕜 F
      G : Type u_4
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace 𝕜 G
      s : Set E
      t : Set F
      q : F → FormalMultilinearSeries 𝕜 F G
      p : E → FormalMultilinearSeries 𝕜 E F
      n : Nat
      c✝ : OrderedFinpartition n
      c : OrderedFinpartition (HAdd.hAdd n 1)
      hc : Ne (Set.range (c.emb 0)) (Singleton.singleton 0)
      i : Fin c.length
      x✝¹ : Membership.mem Set.univ i
      j : Fin c.length
      x✝ : Membership.mem Set.univ j
      hij : Ne i j
      ⊢ Function.onFun Disjoint (fun m => Set.range ((fun i j => dite (Eq i (c.index …
    -/
    wlog h : i ≠ c.index 0 generalizing i j
    · apply Disjoint.symm
        (this j (mem_univ j) i (mem_univ i) hij.symm ?_)
      /-
        𝕜 : Type u_1
        inst✝⁶ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝⁵ : NormedAddCommGroup E
        inst✝⁴ : NormedSpace 𝕜 E
        F : Type u_3
        inst✝³ : NormedAddCommGroup F
        inst✝² : NormedSpace 𝕜 F
        G : Type u_4
        inst✝¹ : NormedAddCommGroup G
        inst✝ : NormedSpace 𝕜 G
        s : Set E
        t : Set F
        q : F → FormalMultilinearSeries 𝕜 F G
        p : E → FormalMultilinearSeries 𝕜 E F
        n : Nat
        c✝ : OrderedFinpartition n
        c : OrderedFinpartition (HAdd.hAdd n 1)
        hc : Ne (Set.range (c.emb 0)) (Singleton.singleton 0)
        i : Fin c.length
        x✝¹ : Membership.mem Set.univ i
        j : Fin c.length
        x✝ : Membership.mem Set.univ j
        hij : Ne i j
        this : ∀ (i : Fin c.length), Membership.mem Set.univ i → ∀ (j : Fin c.length), …
        h : Not (Ne i (c.index 0))
        ⊢ Ne j (c.index 0)
      -/
      simp only [ne_eq, Decidable.not_not] at h
      /-
        𝕜 : Type u_1
        inst✝⁶ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝⁵ : NormedAddCommGroup E
        inst✝⁴ : NormedSpace 𝕜 E
        F : Type u_3
        inst✝³ : NormedAddCommGroup F
        inst✝² : NormedSpace 𝕜 F
        G : Type u_4
        inst✝¹ : NormedAddCommGroup G
        inst✝ : NormedSpace 𝕜 G
        s : Set E
        t : Set F
        q : F → FormalMultilinearSeries 𝕜 F G
        p : E → FormalMultilinearSeries 𝕜 E F
        n : Nat
        c✝ : OrderedFinpartition n
        c : OrderedFinpartition (HAdd.hAdd n 1)
        hc : Ne (Set.range (c.emb 0)) (Singleton.singleton 0)
        i : Fin c.length
        x✝¹ : Membership.mem Set.univ i
        j : Fin c.length
        x✝ : Membership.mem Set.univ j
        hij : Ne i j
        this : ∀ (i : Fin c.length), Membership.mem Set.univ i → ∀ (j : Fin c.length), …
        h : Eq i (c.index 0)
        ⊢ Ne j (c.index 0)
      -/
      simpa [h] using hij.symm
      /-
        🎉 no goals
      -/
    /-
      𝕜 : Type u_1
      inst✝⁶ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace 𝕜 E
      F : Type u_3
      inst✝³ : NormedAddCommGroup F
      inst✝² : NormedSpace 𝕜 F
      G : Type u_4
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace 𝕜 G
      s : Set E
      t : Set F
      q : F → FormalMultilinearSeries 𝕜 F G
      p : E → FormalMultilinearSeries 𝕜 E F
      n : Nat
      c✝ : OrderedFinpartition n
      c : OrderedFinpartition (HAdd.hAdd n 1)
      hc : Ne (Set.range (c.emb 0)) (Singleton.singleton 0)
      i : Fin c.length
      x✝¹ : Membership.mem Set.univ i
      j : Fin c.length
      x✝ : Membership.mem Set.univ j
      hij : Ne i j
      h : Ne i (c.index 0)
      ⊢ Function.onFun Disjoint (fun m => Set.range ((fun i j => dite (Eq i (c.index …
    -/
    rcases eq_or_ne j (c.index 0) with rfl | hj
      /-
        case inl
        𝕜 : Type u_1
        inst✝⁶ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝⁵ : NormedAddCommGroup E
        inst✝⁴ : NormedSpace 𝕜 E
        F : Type u_3
        inst✝³ : NormedAddCommGroup F
        inst✝² : NormedSpace 𝕜 F
        G : Type u_4
        inst✝¹ : NormedAddCommGroup G
        inst✝ : NormedSpace 𝕜 G
        s : Set E
        t : Set F
        q : F → FormalMultilinearSeries 𝕜 F G
        p : E → FormalMultilinearSeries 𝕜 E F
        n : Nat
        c✝ : OrderedFinpartition n
        c : OrderedFinpartition (HAdd.hAdd n 1)
        hc : Ne (Set.range (c.emb 0)) (Singleton.singleton 0)
        i : Fin c.length
        x✝¹ : Membership.mem Set.univ i
        h : Ne i (c.index 0)
        x✝ : Membership.mem Set.univ (c.index 0)
        hij : Ne i (c.index 0)
        ⊢ Function.onFun Disjoint (fun m => Set.range ((fun i j => dite (Eq i (c.index …
      -/
    · simp only [onFun, hij, ↓reduceDIte, Nat.succ_eq_add_one]
      /-
        case inl
        𝕜 : Type u_1
        inst✝⁶ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝⁵ : NormedAddCommGroup E
        inst✝⁴ : NormedSpace 𝕜 E
        F : Type u_3
        inst✝³ : NormedAddCommGroup F
        inst✝² : NormedSpace 𝕜 F
        G : Type u_4
        inst✝¹ : NormedAddCommGroup G
        inst✝ : NormedSpace 𝕜 G
        s : Set E
        t : Set F
        q : F → FormalMultilinearSeries 𝕜 F G
        p : E → FormalMultilinearSeries 𝕜 E F
        n : Nat
        c✝ : OrderedFinpartition n
        c : OrderedFinpartition (HAdd.hAdd n 1)
        hc : Ne (Set.range (c.emb 0)) (Singleton.singleton 0)
        i : Fin c.length
        x✝¹ : Membership.mem Set.univ i
        h : Ne i (c.index 0)
        x✝ : Membership.mem Set.univ (c.index 0)
        hij : Ne i (c.index 0)
        ⊢ Disjoint (Set.range fun j => (c.emb i (Fin.cast ⋯ j)).pred ⋯) (Set.range fun …
      -/
      apply Set.disjoint_iff_forall_ne.2
      /-
        case inl
        𝕜 : Type u_1
        inst✝⁶ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝⁵ : NormedAddCommGroup E
        inst✝⁴ : NormedSpace 𝕜 E
        F : Type u_3
        inst✝³ : NormedAddCommGroup F
        inst✝² : NormedSpace 𝕜 F
        G : Type u_4
        inst✝¹ : NormedAddCommGroup G
        inst✝ : NormedSpace 𝕜 G
        s : Set E
        t : Set F
        q : F → FormalMultilinearSeries 𝕜 F G
        p : E → FormalMultilinearSeries 𝕜 E F
        n : Nat
        c✝ : OrderedFinpartition n
        c : OrderedFinpartition (HAdd.hAdd n 1)
        hc : Ne (Set.range (c.emb 0)) (Singleton.singleton 0)
        i : Fin c.length
        x✝¹ : Membership.mem Set.univ i
        h : Ne i (c.index 0)
        x✝ : Membership.mem Set.univ (c.index 0)
        hij : Ne i (c.index 0)
        ⊢ ∀ ⦃a : Fin n⦄, Membership.mem (Set.range fun j => (c.emb i (Fin.cast ⋯ j)).p …
      -/
      simp only [mem_range, ne_eq, forall_exists_index, forall_apply_eq_imp_iff, pred_inj]
      /-
        case inl
        𝕜 : Type u_1
        inst✝⁶ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝⁵ : NormedAddCommGroup E
        inst✝⁴ : NormedSpace 𝕜 E
        F : Type u_3
        inst✝³ : NormedAddCommGroup F
        inst✝² : NormedSpace 𝕜 F
        G : Type u_4
        inst✝¹ : NormedAddCommGroup G
        inst✝ : NormedSpace 𝕜 G
        s : Set E
        t : Set F
        q : F → FormalMultilinearSeries 𝕜 F G
        p : E → FormalMultilinearSeries 𝕜 E F
        n : Nat
        c✝ : OrderedFinpartition n
        c : OrderedFinpartition (HAdd.hAdd n 1)
        hc : Ne (Set.range (c.emb 0)) (Singleton.singleton 0)
        i : Fin c.length
        x✝¹ : Membership.mem Set.univ i
        h : Ne i (c.index 0)
        x✝ : Membership.mem Set.univ (c.index 0)
        hij : Ne i (c.index 0)
        ⊢ ∀ (a : Fin (Function.update c.partSize (c.index 0) (HSub.hSub (c.partSize (c …
      -/
      intro a b
      /-
        case inl
        𝕜 : Type u_1
        inst✝⁶ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝⁵ : NormedAddCommGroup E
        inst✝⁴ : NormedSpace 𝕜 E
        F : Type u_3
        inst✝³ : NormedAddCommGroup F
        inst✝² : NormedSpace 𝕜 F
        G : Type u_4
        inst✝¹ : NormedAddCommGroup G
        inst✝ : NormedSpace 𝕜 G
        s : Set E
        t : Set F
        q : F → FormalMultilinearSeries 𝕜 F G
        p : E → FormalMultilinearSeries 𝕜 E F
        n : Nat
        c✝ : OrderedFinpartition n
        c : OrderedFinpartition (HAdd.hAdd n 1)
        hc : Ne (Set.range (c.emb 0)) (Singleton.singleton 0)
        i : Fin c.length
        x✝¹ : Membership.mem Set.univ i
        h : Ne i (c.index 0)
        x✝ : Membership.mem Set.univ (c.index 0)
        hij : Ne i (c.index 0)
        a : Fin (Function.update c.partSize (c.index 0) (HSub.hSub (c.partSize (c.inde …
        b : Fin (Function.update c.partSize (c.index 0) (HSub.hSub (c.partSize (c.inde …
        ⊢ Not (Eq (c.emb i (Fin.cast ⋯ a)) (c.emb (c.index 0) (Fin.cast ⋯ b.succ)))
      -/
      exact c.emb_ne_emb_of_ne hij
      /-
        🎉 no goals
      -/
      /-
        case inr
        𝕜 : Type u_1
        inst✝⁶ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝⁵ : NormedAddCommGroup E
        inst✝⁴ : NormedSpace 𝕜 E
        F : Type u_3
        inst✝³ : NormedAddCommGroup F
        inst✝² : NormedSpace 𝕜 F
        G : Type u_4
        inst✝¹ : NormedAddCommGroup G
        inst✝ : NormedSpace 𝕜 G
        s : Set E
        t : Set F
        q : F → FormalMultilinearSeries 𝕜 F G
        p : E → FormalMultilinearSeries 𝕜 E F
        n : Nat
        c✝ : OrderedFinpartition n
        c : OrderedFinpartition (HAdd.hAdd n 1)
        hc : Ne (Set.range (c.emb 0)) (Singleton.singleton 0)
        i : Fin c.length
        x✝¹ : Membership.mem Set.univ i
        j : Fin c.length
        x✝ : Membership.mem Set.univ j
        hij : Ne i j
        h : Ne i (c.index 0)
        hj : Ne j (c.index 0)
        ⊢ Function.onFun Disjoint (fun m => Set.range ((fun i j => dite (Eq i (c.index …
      -/
    · simp only [onFun, h, ↓reduceDIte, hj]
      /-
        case inr
        𝕜 : Type u_1
        inst✝⁶ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝⁵ : NormedAddCommGroup E
        inst✝⁴ : NormedSpace 𝕜 E
        F : Type u_3
        inst✝³ : NormedAddCommGroup F
        inst✝² : NormedSpace 𝕜 F
        G : Type u_4
        inst✝¹ : NormedAddCommGroup G
        inst✝ : NormedSpace 𝕜 G
        s : Set E
        t : Set F
        q : F → FormalMultilinearSeries 𝕜 F G
        p : E → FormalMultilinearSeries 𝕜 E F
        n : Nat
        c✝ : OrderedFinpartition n
        c : OrderedFinpartition (HAdd.hAdd n 1)
        hc : Ne (Set.range (c.emb 0)) (Singleton.singleton 0)
        i : Fin c.length
        x✝¹ : Membership.mem Set.univ i
        j : Fin c.length
        x✝ : Membership.mem Set.univ j
        hij : Ne i j
        h : Ne i (c.index 0)
        hj : Ne j (c.index 0)
        ⊢ Disjoint (Set.range fun j => (c.emb i (Fin.cast ⋯ j)).pred ⋯) (Set.range fun …
      -/
      apply Set.disjoint_iff_forall_ne.2
      /-
        case inr
        𝕜 : Type u_1
        inst✝⁶ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝⁵ : NormedAddCommGroup E
        inst✝⁴ : NormedSpace 𝕜 E
        F : Type u_3
        inst✝³ : NormedAddCommGroup F
        inst✝² : NormedSpace 𝕜 F
        G : Type u_4
        inst✝¹ : NormedAddCommGroup G
        inst✝ : NormedSpace 𝕜 G
        s : Set E
        t : Set F
        q : F → FormalMultilinearSeries 𝕜 F G
        p : E → FormalMultilinearSeries 𝕜 E F
        n : Nat
        c✝ : OrderedFinpartition n
        c : OrderedFinpartition (HAdd.hAdd n 1)
        hc : Ne (Set.range (c.emb 0)) (Singleton.singleton 0)
        i : Fin c.length
        x✝¹ : Membership.mem Set.univ i
        j : Fin c.length
        x✝ : Membership.mem Set.univ j
        hij : Ne i j
        h : Ne i (c.index 0)
        hj : Ne j (c.index 0)
        ⊢ ∀ ⦃a : Fin n⦄, Membership.mem (Set.range fun j => (c.emb i (Fin.cast ⋯ j)).p …
      -/
      simp only [mem_range, ne_eq, forall_exists_index, forall_apply_eq_imp_iff, pred_inj]
      /-
        case inr
        𝕜 : Type u_1
        inst✝⁶ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝⁵ : NormedAddCommGroup E
        inst✝⁴ : NormedSpace 𝕜 E
        F : Type u_3
        inst✝³ : NormedAddCommGroup F
        inst✝² : NormedSpace 𝕜 F
        G : Type u_4
        inst✝¹ : NormedAddCommGroup G
        inst✝ : NormedSpace 𝕜 G
        s : Set E
        t : Set F
        q : F → FormalMultilinearSeries 𝕜 F G
        p : E → FormalMultilinearSeries 𝕜 E F
        n : Nat
        c✝ : OrderedFinpartition n
        c : OrderedFinpartition (HAdd.hAdd n 1)
        hc : Ne (Set.range (c.emb 0)) (Singleton.singleton 0)
        i : Fin c.length
        x✝¹ : Membership.mem Set.univ i
        j : Fin c.length
        x✝ : Membership.mem Set.univ j
        hij : Ne i j
        h : Ne i (c.index 0)
        hj : Ne j (c.index 0)
        ⊢ ∀ (a : Fin (Function.update c.partSize (c.index 0) (HSub.hSub (c.partSize (c …
      -/
      intro a b
      /-
        case inr
        𝕜 : Type u_1
        inst✝⁶ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝⁵ : NormedAddCommGroup E
        inst✝⁴ : NormedSpace 𝕜 E
        F : Type u_3
        inst✝³ : NormedAddCommGroup F
        inst✝² : NormedSpace 𝕜 F
        G : Type u_4
        inst✝¹ : NormedAddCommGroup G
        inst✝ : NormedSpace 𝕜 G
        s : Set E
        t : Set F
        q : F → FormalMultilinearSeries 𝕜 F G
        p : E → FormalMultilinearSeries 𝕜 E F
        n : Nat
        c✝ : OrderedFinpartition n
        c : OrderedFinpartition (HAdd.hAdd n 1)
        hc : Ne (Set.range (c.emb 0)) (Singleton.singleton 0)
        i : Fin c.length
        x✝¹ : Membership.mem Set.univ i
        j : Fin c.length
        x✝ : Membership.mem Set.univ j
        hij : Ne i j
        h : Ne i (c.index 0)
        hj : Ne j (c.index 0)
        a : Fin (Function.update c.partSize (c.index 0) (HSub.hSub (c.partSize (c.inde …
        b : Fin (Function.update c.partSize (c.index 0) (HSub.hSub (c.partSize (c.inde …
        ⊢ Not (Eq (c.emb i (Fin.cast ⋯ a)) (c.emb j (Fin.cast ⋯ b)))
      -/
      exact c.emb_ne_emb_of_ne hij
      /-
        🎉 no goals
      -/
  cover x := by
    /-
      𝕜 : Type u_1
      inst✝⁶ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace 𝕜 E
      F : Type u_3
      inst✝³ : NormedAddCommGroup F
      inst✝² : NormedSpace 𝕜 F
      G : Type u_4
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace 𝕜 G
      s : Set E
      t : Set F
      q : F → FormalMultilinearSeries 𝕜 F G
      p : E → FormalMultilinearSeries 𝕜 E F
      n : Nat
      c✝ : OrderedFinpartition n
      c : OrderedFinpartition (HAdd.hAdd n 1)
      hc : Ne (Set.range (c.emb 0)) (Singleton.singleton 0)
      x : Fin n
      ⊢ Exists fun m => Membership.mem (Set.range ((fun i j => dite (Eq i (c.index 0 …
    -/
    simp only [mem_iUnion, mem_range]
    obtain ⟨i, j, hij⟩ : ∃ (i : Fin c.length), ∃ (j : Fin (c.partSize i)), c.emb i j = succ x :=
      ⟨c.index (succ x), c.invEmbedding (succ x), by simp⟩
    /-
      case intro.intro
      𝕜 : Type u_1
      inst✝⁶ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace 𝕜 E
      F : Type u_3
      inst✝³ : NormedAddCommGroup F
      inst✝² : NormedSpace 𝕜 F
      G : Type u_4
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace 𝕜 G
      s : Set E
      t : Set F
      q : F → FormalMultilinearSeries 𝕜 F G
      p : E → FormalMultilinearSeries 𝕜 E F
      n : Nat
      c✝ : OrderedFinpartition n
      c : OrderedFinpartition (HAdd.hAdd n 1)
      hc : Ne (Set.range (c.emb 0)) (Singleton.singleton 0)
      x : Fin n
      i : Fin c.length
      j : Fin (c.partSize i)
      hij : Eq (c.emb i j) x.succ
      ⊢ Exists fun m => Exists fun y => Eq (dite (Eq m (c.index 0)) (fun h => (c.emb …
    -/
    rcases eq_or_ne i (c.index 0) with rfl | hi
      /-
        case intro.intro.inl
        𝕜 : Type u_1
        inst✝⁶ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝⁵ : NormedAddCommGroup E
        inst✝⁴ : NormedSpace 𝕜 E
        F : Type u_3
        inst✝³ : NormedAddCommGroup F
        inst✝² : NormedSpace 𝕜 F
        G : Type u_4
        inst✝¹ : NormedAddCommGroup G
        inst✝ : NormedSpace 𝕜 G
        s : Set E
        t : Set F
        q : F → FormalMultilinearSeries 𝕜 F G
        p : E → FormalMultilinearSeries 𝕜 E F
        n : Nat
        c✝ : OrderedFinpartition n
        c : OrderedFinpartition (HAdd.hAdd n 1)
        hc : Ne (Set.range (c.emb 0)) (Singleton.singleton 0)
        x : Fin n
        j : Fin (c.partSize (c.index 0))
        hij : Eq (c.emb (c.index 0) j) x.succ
        ⊢ Exists fun m => Exists fun y => Eq (dite (Eq m (c.index 0)) (fun h => (c.emb …
      -/
    · refine ⟨c.index 0, ?_⟩
      have j_ne : j ≠ 0 := by
        rintro rfl
        simp only [c.emb_zero] at hij
        exact (Fin.succ_ne_zero _).symm hij
      /-
        case intro.intro.inl
        𝕜 : Type u_1
        inst✝⁶ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝⁵ : NormedAddCommGroup E
        inst✝⁴ : NormedSpace 𝕜 E
        F : Type u_3
        inst✝³ : NormedAddCommGroup F
        inst✝² : NormedSpace 𝕜 F
        G : Type u_4
        inst✝¹ : NormedAddCommGroup G
        inst✝ : NormedSpace 𝕜 G
        s : Set E
        t : Set F
        q : F → FormalMultilinearSeries 𝕜 F G
        p : E → FormalMultilinearSeries 𝕜 E F
        n : Nat
        c✝ : OrderedFinpartition n
        c : OrderedFinpartition (HAdd.hAdd n 1)
        hc : Ne (Set.range (c.emb 0)) (Singleton.singleton 0)
        x : Fin n
        j : Fin (c.partSize (c.index 0))
        hij : Eq (c.emb (c.index 0) j) x.succ
        j_ne : Ne j 0
        ⊢ Exists fun y => Eq (dite (Eq (c.index 0) (c.index 0)) (fun h => (c.emb (c.in …
      -/
      have je_ne' : (j : ℕ) ≠ 0 := by simpa [← val_eq_val] using j_ne
      /-
        case intro.intro.inl
        𝕜 : Type u_1
        inst✝⁶ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝⁵ : NormedAddCommGroup E
        inst✝⁴ : NormedSpace 𝕜 E
        F : Type u_3
        inst✝³ : NormedAddCommGroup F
        inst✝² : NormedSpace 𝕜 F
        G : Type u_4
        inst✝¹ : NormedAddCommGroup G
        inst✝ : NormedSpace 𝕜 G
        s : Set E
        t : Set F
        q : F → FormalMultilinearSeries 𝕜 F G
        p : E → FormalMultilinearSeries 𝕜 E F
        n : Nat
        c✝ : OrderedFinpartition n
        c : OrderedFinpartition (HAdd.hAdd n 1)
        hc : Ne (Set.range (c.emb 0)) (Singleton.singleton 0)
        x : Fin n
        j : Fin (c.partSize (c.index 0))
        hij : Eq (c.emb (c.index 0) j) x.succ
        j_ne : Ne j 0
        je_ne' : Ne (↑j) 0
        ⊢ Exists fun y => Eq (dite (Eq (c.index 0) (c.index 0)) (fun h => (c.emb (c.in …
      -/
      simp only [↓reduceDIte, Nat.succ_eq_add_one]
      have A : c.partSize (c.index 0) - 1 + 1 = c.partSize (c.index 0) :=
        Nat.sub_add_cancel (c.partSize_pos _)
      have B : update c.partSize (c.index 0) (c.partSize (c.index 0) - 1) (c.index 0) =
        c.partSize (c.index 0) - 1 := by simp
      /-
        case intro.intro.inl
        𝕜 : Type u_1
        inst✝⁶ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝⁵ : NormedAddCommGroup E
        inst✝⁴ : NormedSpace 𝕜 E
        F : Type u_3
        inst✝³ : NormedAddCommGroup F
        inst✝² : NormedSpace 𝕜 F
        G : Type u_4
        inst✝¹ : NormedAddCommGroup G
        inst✝ : NormedSpace 𝕜 G
        s : Set E
        t : Set F
        q : F → FormalMultilinearSeries 𝕜 F G
        p : E → FormalMultilinearSeries 𝕜 E F
        n : Nat
        c✝ : OrderedFinpartition n
        c : OrderedFinpartition (HAdd.hAdd n 1)
        hc : Ne (Set.range (c.emb 0)) (Singleton.singleton 0)
        x : Fin n
        j : Fin (c.partSize (c.index 0))
        hij : Eq (c.emb (c.index 0) j) x.succ
        j_ne : Ne j 0
        je_ne' : Ne (↑j) 0
        A : Eq (HAdd.hAdd (HSub.hSub (c.partSize (c.index 0)) 1) 1) (c.partSize (c.ind …
        B : Eq (Function.update c.partSize (c.index 0) (HSub.hSub (c.partSize (c.index …
        ⊢ Exists fun y => Eq ((c.emb (c.index 0) (Fin.cast ⋯ y.succ)).pred ⋯) x
      -/
      refine ⟨Fin.cast B.symm (pred (Fin.cast A.symm j) ?_), ?_⟩
        /-
          case intro.intro.inl.refine_1
          𝕜 : Type u_1
          inst✝⁶ : NontriviallyNormedField 𝕜
          E : Type u_2
          inst✝⁵ : NormedAddCommGroup E
          inst✝⁴ : NormedSpace 𝕜 E
          F : Type u_3
          inst✝³ : NormedAddCommGroup F
          inst✝² : NormedSpace 𝕜 F
          G : Type u_4
          inst✝¹ : NormedAddCommGroup G
          inst✝ : NormedSpace 𝕜 G
          s : Set E
          t : Set F
          q : F → FormalMultilinearSeries 𝕜 F G
          p : E → FormalMultilinearSeries 𝕜 E F
          n : Nat
          c✝ : OrderedFinpartition n
          c : OrderedFinpartition (HAdd.hAdd n 1)
          hc : Ne (Set.range (c.emb 0)) (Singleton.singleton 0)
          x : Fin n
          j : Fin (c.partSize (c.index 0))
          hij : Eq (c.emb (c.index 0) j) x.succ
          j_ne : Ne j 0
          je_ne' : Ne (↑j) 0
          A : Eq (HAdd.hAdd (HSub.hSub (c.partSize (c.index 0)) 1) 1) (c.partSize (c.ind …
          B : Eq (Function.update c.partSize (c.index 0) (HSub.hSub (c.partSize (c.index …
          ⊢ Ne (Fin.cast ⋯ j) 0
        -/
      · simpa using j_ne
        /-
          🎉 no goals
        -/
        /-
          case intro.intro.inl.refine_2
          𝕜 : Type u_1
          inst✝⁶ : NontriviallyNormedField 𝕜
          E : Type u_2
          inst✝⁵ : NormedAddCommGroup E
          inst✝⁴ : NormedSpace 𝕜 E
          F : Type u_3
          inst✝³ : NormedAddCommGroup F
          inst✝² : NormedSpace 𝕜 F
          G : Type u_4
          inst✝¹ : NormedAddCommGroup G
          inst✝ : NormedSpace 𝕜 G
          s : Set E
          t : Set F
          q : F → FormalMultilinearSeries 𝕜 F G
          p : E → FormalMultilinearSeries 𝕜 E F
          n : Nat
          c✝ : OrderedFinpartition n
          c : OrderedFinpartition (HAdd.hAdd n 1)
          hc : Ne (Set.range (c.emb 0)) (Singleton.singleton 0)
          x : Fin n
          j : Fin (c.partSize (c.index 0))
          hij : Eq (c.emb (c.index 0) j) x.succ
          j_ne : Ne j 0
          je_ne' : Ne (↑j) 0
          A : Eq (HAdd.hAdd (HSub.hSub (c.partSize (c.index 0)) 1) 1) (c.partSize (c.ind …
          B : Eq (Function.update c.partSize (c.index 0) (HSub.hSub (c.partSize (c.index …
          ⊢ Eq ((c.emb (c.index 0) (Fin.cast ⋯ (Fin.cast ⋯ ((Fin.cast ⋯ j).pred ⋯)).succ …
        -/
      · have : x = pred (succ x) (succ_ne_zero x) := rfl
        /-
          case intro.intro.inl.refine_2
          𝕜 : Type u_1
          inst✝⁶ : NontriviallyNormedField 𝕜
          E : Type u_2
          inst✝⁵ : NormedAddCommGroup E
          inst✝⁴ : NormedSpace 𝕜 E
          F : Type u_3
          inst✝³ : NormedAddCommGroup F
          inst✝² : NormedSpace 𝕜 F
          G : Type u_4
          inst✝¹ : NormedAddCommGroup G
          inst✝ : NormedSpace 𝕜 G
          s : Set E
          t : Set F
          q : F → FormalMultilinearSeries 𝕜 F G
          p : E → FormalMultilinearSeries 𝕜 E F
          n : Nat
          c✝ : OrderedFinpartition n
          c : OrderedFinpartition (HAdd.hAdd n 1)
          hc : Ne (Set.range (c.emb 0)) (Singleton.singleton 0)
          x : Fin n
          j : Fin (c.partSize (c.index 0))
          hij : Eq (c.emb (c.index 0) j) x.succ
          j_ne : Ne j 0
          je_ne' : Ne (↑j) 0
          A : Eq (HAdd.hAdd (HSub.hSub (c.partSize (c.index 0)) 1) 1) (c.partSize (c.ind …
          B : Eq (Function.update c.partSize (c.index 0) (HSub.hSub (c.partSize (c.index …
          this : Eq x (x.succ.pred ⋯)
          ⊢ Eq ((c.emb (c.index 0) (Fin.cast ⋯ (Fin.cast ⋯ ((Fin.cast ⋯ j).pred ⋯)).succ …
        -/
        rw [this]
        /-
          case intro.intro.inl.refine_2
          𝕜 : Type u_1
          inst✝⁶ : NontriviallyNormedField 𝕜
          E : Type u_2
          inst✝⁵ : NormedAddCommGroup E
          inst✝⁴ : NormedSpace 𝕜 E
          F : Type u_3
          inst✝³ : NormedAddCommGroup F
          inst✝² : NormedSpace 𝕜 F
          G : Type u_4
          inst✝¹ : NormedAddCommGroup G
          inst✝ : NormedSpace 𝕜 G
          s : Set E
          t : Set F
          q : F → FormalMultilinearSeries 𝕜 F G
          p : E → FormalMultilinearSeries 𝕜 E F
          n : Nat
          c✝ : OrderedFinpartition n
          c : OrderedFinpartition (HAdd.hAdd n 1)
          hc : Ne (Set.range (c.emb 0)) (Singleton.singleton 0)
          x : Fin n
          j : Fin (c.partSize (c.index 0))
          hij : Eq (c.emb (c.index 0) j) x.succ
          j_ne : Ne j 0
          je_ne' : Ne (↑j) 0
          A : Eq (HAdd.hAdd (HSub.hSub (c.partSize (c.index 0)) 1) 1) (c.partSize (c.ind …
          B : Eq (Function.update c.partSize (c.index 0) (HSub.hSub (c.partSize (c.index …
          this : Eq x (x.succ.pred ⋯)
          ⊢ Eq ((c.emb (c.index 0) (Fin.cast ⋯ (Fin.cast ⋯ ((Fin.cast ⋯ j).pred ⋯)).succ …
        -/
        simp only [pred_inj, ← hij]
        /-
          case intro.intro.inl.refine_2
          𝕜 : Type u_1
          inst✝⁶ : NontriviallyNormedField 𝕜
          E : Type u_2
          inst✝⁵ : NormedAddCommGroup E
          inst✝⁴ : NormedSpace 𝕜 E
          F : Type u_3
          inst✝³ : NormedAddCommGroup F
          inst✝² : NormedSpace 𝕜 F
          G : Type u_4
          inst✝¹ : NormedAddCommGroup G
          inst✝ : NormedSpace 𝕜 G
          s : Set E
          t : Set F
          q : F → FormalMultilinearSeries 𝕜 F G
          p : E → FormalMultilinearSeries 𝕜 E F
          n : Nat
          c✝ : OrderedFinpartition n
          c : OrderedFinpartition (HAdd.hAdd n 1)
          hc : Ne (Set.range (c.emb 0)) (Singleton.singleton 0)
          x : Fin n
          j : Fin (c.partSize (c.index 0))
          hij : Eq (c.emb (c.index 0) j) x.succ
          j_ne : Ne j 0
          je_ne' : Ne (↑j) 0
          A : Eq (HAdd.hAdd (HSub.hSub (c.partSize (c.index 0)) 1) 1) (c.partSize (c.ind …
          B : Eq (Function.update c.partSize (c.index 0) (HSub.hSub (c.partSize (c.index …
          this : Eq x (x.succ.pred ⋯)
          ⊢ Eq (c.emb (c.index 0) (Fin.cast ⋯ (Fin.cast ⋯ ((Fin.cast ⋯ j).pred ⋯)).succ) …
        -/
        congr 1
        /-
          case intro.intro.inl.refine_2.e_a
          𝕜 : Type u_1
          inst✝⁶ : NontriviallyNormedField 𝕜
          E : Type u_2
          inst✝⁵ : NormedAddCommGroup E
          inst✝⁴ : NormedSpace 𝕜 E
          F : Type u_3
          inst✝³ : NormedAddCommGroup F
          inst✝² : NormedSpace 𝕜 F
          G : Type u_4
          inst✝¹ : NormedAddCommGroup G
          inst✝ : NormedSpace 𝕜 G
          s : Set E
          t : Set F
          q : F → FormalMultilinearSeries 𝕜 F G
          p : E → FormalMultilinearSeries 𝕜 E F
          n : Nat
          c✝ : OrderedFinpartition n
          c : OrderedFinpartition (HAdd.hAdd n 1)
          hc : Ne (Set.range (c.emb 0)) (Singleton.singleton 0)
          x : Fin n
          j : Fin (c.partSize (c.index 0))
          hij : Eq (c.emb (c.index 0) j) x.succ
          j_ne : Ne j 0
          je_ne' : Ne (↑j) 0
          A : Eq (HAdd.hAdd (HSub.hSub (c.partSize (c.index 0)) 1) 1) (c.partSize (c.ind …
          B : Eq (Function.update c.partSize (c.index 0) (HSub.hSub (c.partSize (c.index …
          this : Eq x (x.succ.pred ⋯)
          ⊢ Eq (Fin.cast ⋯ (Fin.cast ⋯ ((Fin.cast ⋯ j).pred ⋯)).succ) j
        -/
        rw [← val_eq_val]
        /-
          case intro.intro.inl.refine_2.e_a
          𝕜 : Type u_1
          inst✝⁶ : NontriviallyNormedField 𝕜
          E : Type u_2
          inst✝⁵ : NormedAddCommGroup E
          inst✝⁴ : NormedSpace 𝕜 E
          F : Type u_3
          inst✝³ : NormedAddCommGroup F
          inst✝² : NormedSpace 𝕜 F
          G : Type u_4
          inst✝¹ : NormedAddCommGroup G
          inst✝ : NormedSpace 𝕜 G
          s : Set E
          t : Set F
          q : F → FormalMultilinearSeries 𝕜 F G
          p : E → FormalMultilinearSeries 𝕜 E F
          n : Nat
          c✝ : OrderedFinpartition n
          c : OrderedFinpartition (HAdd.hAdd n 1)
          hc : Ne (Set.range (c.emb 0)) (Singleton.singleton 0)
          x : Fin n
          j : Fin (c.partSize (c.index 0))
          hij : Eq (c.emb (c.index 0) j) x.succ
          j_ne : Ne j 0
          je_ne' : Ne (↑j) 0
          A : Eq (HAdd.hAdd (HSub.hSub (c.partSize (c.index 0)) 1) 1) (c.partSize (c.ind …
          B : Eq (Function.update c.partSize (c.index 0) (HSub.hSub (c.partSize (c.index …
          this : Eq x (x.succ.pred ⋯)
          ⊢ Eq ↑(Fin.cast ⋯ (Fin.cast ⋯ ((Fin.cast ⋯ j).pred ⋯)).succ) ↑j
        -/
        simp only [coe_cast, val_succ, coe_pred]
        /-
          case intro.intro.inl.refine_2.e_a
          𝕜 : Type u_1
          inst✝⁶ : NontriviallyNormedField 𝕜
          E : Type u_2
          inst✝⁵ : NormedAddCommGroup E
          inst✝⁴ : NormedSpace 𝕜 E
          F : Type u_3
          inst✝³ : NormedAddCommGroup F
          inst✝² : NormedSpace 𝕜 F
          G : Type u_4
          inst✝¹ : NormedAddCommGroup G
          inst✝ : NormedSpace 𝕜 G
          s : Set E
          t : Set F
          q : F → FormalMultilinearSeries 𝕜 F G
          p : E → FormalMultilinearSeries 𝕜 E F
          n : Nat
          c✝ : OrderedFinpartition n
          c : OrderedFinpartition (HAdd.hAdd n 1)
          hc : Ne (Set.range (c.emb 0)) (Singleton.singleton 0)
          x : Fin n
          j : Fin (c.partSize (c.index 0))
          hij : Eq (c.emb (c.index 0) j) x.succ
          j_ne : Ne j 0
          je_ne' : Ne (↑j) 0
          A : Eq (HAdd.hAdd (HSub.hSub (c.partSize (c.index 0)) 1) 1) (c.partSize (c.ind …
          B : Eq (Function.update c.partSize (c.index 0) (HSub.hSub (c.partSize (c.index …
          this : Eq x (x.succ.pred ⋯)
          ⊢ Eq (HAdd.hAdd (HSub.hSub (↑j) 1) 1) ↑j
        -/
        omega
        /-
          🎉 no goals
        -/
    · have A : update c.partSize (c.index 0) (c.partSize (c.index 0) - 1) i = c.partSize i := by
        simp [hi]
      /-
        case intro.intro.inr
        𝕜 : Type u_1
        inst✝⁶ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝⁵ : NormedAddCommGroup E
        inst✝⁴ : NormedSpace 𝕜 E
        F : Type u_3
        inst✝³ : NormedAddCommGroup F
        inst✝² : NormedSpace 𝕜 F
        G : Type u_4
        inst✝¹ : NormedAddCommGroup G
        inst✝ : NormedSpace 𝕜 G
        s : Set E
        t : Set F
        q : F → FormalMultilinearSeries 𝕜 F G
        p : E → FormalMultilinearSeries 𝕜 E F
        n : Nat
        c✝ : OrderedFinpartition n
        c : OrderedFinpartition (HAdd.hAdd n 1)
        hc : Ne (Set.range (c.emb 0)) (Singleton.singleton 0)
        x : Fin n
        i : Fin c.length
        j : Fin (c.partSize i)
        hij : Eq (c.emb i j) x.succ
        hi : Ne i (c.index 0)
        A : Eq (Function.update c.partSize (c.index 0) (HSub.hSub (c.partSize (c.index …
        ⊢ Exists fun m => Exists fun y => Eq (dite (Eq m (c.index 0)) (fun h => (c.emb …
      -/
      exact ⟨i, Fin.cast A.symm j, by simp [hi, hij]⟩
      /-
        🎉 no goals
      -/


open Classical in
/-- Extending the ordered partitions of `Fin n` bijects with the ordered partitions
of `Fin (n+1)`. -/
def extendEquiv (n : ℕ) :
    ((c : OrderedFinpartition n) × Option (Fin c.length)) ≃ OrderedFinpartition (n + 1) where
  toFun c := c.1.extend c.2
  invFun c := if h : range (c.emb 0) = {0} then ⟨c.eraseLeft h, none⟩ else
    ⟨c.eraseMiddle h, some (c.index 0)⟩
  left_inv := by
    /-
      𝕜 : Type u_1
      inst✝⁶ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace 𝕜 E
      F : Type u_3
      inst✝³ : NormedAddCommGroup F
      inst✝² : NormedSpace 𝕜 F
      G : Type u_4
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace 𝕜 G
      s : Set E
      t : Set F
      q : F → FormalMultilinearSeries 𝕜 F G
      p : E → FormalMultilinearSeries 𝕜 E F
      n✝ : Nat
      c : OrderedFinpartition n✝
      n : Nat
      ⊢ Function.LeftInverse (fun c => dite (Eq (Set.range (c.emb 0)) (Singleton.sin …
    -/
    rintro ⟨c, o⟩
    match o with
    | none =>
      simp only [extend, range_extendLeft_zero, ↓reduceDIte, Sigma.mk.inj_iff, heq_eq_eq,
        and_true]
      rfl
    | some i =>
      simp only [extend, range_emb_extendMiddle_ne_singleton_zero, ↓reduceDIte,
        Sigma.mk.inj_iff, heq_eq_eq, and_true, eraseMiddle, Nat.succ_eq_add_one,
        index_extendMiddle_zero]
      ext
      · rfl
      · simp only [Nat.succ_eq_add_one, ne_eq, id_eq, heq_eq_eq, index_extendMiddle_zero]
        ext j
        rcases eq_or_ne i j with rfl | hij
        · simp [extendMiddle]
        · simp [hij.symm, extendMiddle]
      · refine HEq.symm (hfunext rfl ?_)
        simp only [Nat.succ_eq_add_one, heq_eq_eq, forall_eq']
        intro a
        rcases eq_or_ne a i with rfl | hij
        · refine (Fin.heq_fun_iff ?_).mpr ?_
          · rw [index_extendMiddle_zero]
            simp [extendMiddle]
          · simp [extendMiddle]
        · refine (Fin.heq_fun_iff ?_).mpr ?_
          · rw [index_extendMiddle_zero]
            simp [extendMiddle]
          · simp [extendMiddle, hij]
  right_inv c := by
    /-
      𝕜 : Type u_1
      inst✝⁶ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace 𝕜 E
      F : Type u_3
      inst✝³ : NormedAddCommGroup F
      inst✝² : NormedSpace 𝕜 F
      G : Type u_4
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace 𝕜 G
      s : Set E
      t : Set F
      q : F → FormalMultilinearSeries 𝕜 F G
      p : E → FormalMultilinearSeries 𝕜 E F
      n✝ : Nat
      c✝ : OrderedFinpartition n✝
      n : Nat
      c : OrderedFinpartition (HAdd.hAdd n 1)
      ⊢ Eq ((fun c => c.fst.extend c.snd) ((fun c => dite (Eq (Set.range (c.emb 0))  …
    -/
    by_cases h : range (c.emb 0) = {0}
      /-
        case pos
        𝕜 : Type u_1
        inst✝⁶ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝⁵ : NormedAddCommGroup E
        inst✝⁴ : NormedSpace 𝕜 E
        F : Type u_3
        inst✝³ : NormedAddCommGroup F
        inst✝² : NormedSpace 𝕜 F
        G : Type u_4
        inst✝¹ : NormedAddCommGroup G
        inst✝ : NormedSpace 𝕜 G
        s : Set E
        t : Set F
        q : F → FormalMultilinearSeries 𝕜 F G
        p : E → FormalMultilinearSeries 𝕜 E F
        n✝ : Nat
        c✝ : OrderedFinpartition n✝
        n : Nat
        c : OrderedFinpartition (HAdd.hAdd n 1)
        h : Eq (Set.range (c.emb 0)) (Singleton.singleton 0)
        ⊢ Eq ((fun c => c.fst.extend c.snd) ((fun c => dite (Eq (Set.range (c.emb 0))  …
      -/
    · have A : c.length - 1 + 1 = c.length := Nat.sub_add_cancel (c.length_pos (Nat.zero_lt_succ n))
      /-
        case pos
        𝕜 : Type u_1
        inst✝⁶ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝⁵ : NormedAddCommGroup E
        inst✝⁴ : NormedSpace 𝕜 E
        F : Type u_3
        inst✝³ : NormedAddCommGroup F
        inst✝² : NormedSpace 𝕜 F
        G : Type u_4
        inst✝¹ : NormedAddCommGroup G
        inst✝ : NormedSpace 𝕜 G
        s : Set E
        t : Set F
        q : F → FormalMultilinearSeries 𝕜 F G
        p : E → FormalMultilinearSeries 𝕜 E F
        n✝ : Nat
        c✝ : OrderedFinpartition n✝
        n : Nat
        c : OrderedFinpartition (HAdd.hAdd n 1)
        h : Eq (Set.range (c.emb 0)) (Singleton.singleton 0)
        A : Eq (HAdd.hAdd (HSub.hSub c.length 1) 1) c.length
        ⊢ Eq ((fun c => c.fst.extend c.snd) ((fun c => dite (Eq (Set.range (c.emb 0))  …
      -/
      dsimp only
      /-
        case pos
        𝕜 : Type u_1
        inst✝⁶ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝⁵ : NormedAddCommGroup E
        inst✝⁴ : NormedSpace 𝕜 E
        F : Type u_3
        inst✝³ : NormedAddCommGroup F
        inst✝² : NormedSpace 𝕜 F
        G : Type u_4
        inst✝¹ : NormedAddCommGroup G
        inst✝ : NormedSpace 𝕜 G
        s : Set E
        t : Set F
        q : F → FormalMultilinearSeries 𝕜 F G
        p : E → FormalMultilinearSeries 𝕜 E F
        n✝ : Nat
        c✝ : OrderedFinpartition n✝
        n : Nat
        c : OrderedFinpartition (HAdd.hAdd n 1)
        h : Eq (Set.range (c.emb 0)) (Singleton.singleton 0)
        A : Eq (HAdd.hAdd (HSub.hSub c.length 1) 1) c.length
        ⊢ Eq ((dite (Eq (Set.range (c.emb 0)) (Singleton.singleton 0)) (fun h => ⟨c.er …
      -/
      rw [dif_pos h]
      /-
        case pos
        𝕜 : Type u_1
        inst✝⁶ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝⁵ : NormedAddCommGroup E
        inst✝⁴ : NormedSpace 𝕜 E
        F : Type u_3
        inst✝³ : NormedAddCommGroup F
        inst✝² : NormedSpace 𝕜 F
        G : Type u_4
        inst✝¹ : NormedAddCommGroup G
        inst✝ : NormedSpace 𝕜 G
        s : Set E
        t : Set F
        q : F → FormalMultilinearSeries 𝕜 F G
        p : E → FormalMultilinearSeries 𝕜 E F
        n✝ : Nat
        c✝ : OrderedFinpartition n✝
        n : Nat
        c : OrderedFinpartition (HAdd.hAdd n 1)
        h : Eq (Set.range (c.emb 0)) (Singleton.singleton 0)
        A : Eq (HAdd.hAdd (HSub.hSub c.length 1) 1) c.length
        ⊢ Eq (⟨c.eraseLeft h, Option.none⟩.fst.extend ⟨c.eraseLeft h, Option.none⟩.snd …
      -/
      simp only [extend, extendLeft, eraseLeft]
      /-
        case pos
        𝕜 : Type u_1
        inst✝⁶ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝⁵ : NormedAddCommGroup E
        inst✝⁴ : NormedSpace 𝕜 E
        F : Type u_3
        inst✝³ : NormedAddCommGroup F
        inst✝² : NormedSpace 𝕜 F
        G : Type u_4
        inst✝¹ : NormedAddCommGroup G
        inst✝ : NormedSpace 𝕜 G
        s : Set E
        t : Set F
        q : F → FormalMultilinearSeries 𝕜 F G
        p : E → FormalMultilinearSeries 𝕜 E F
        n✝ : Nat
        c✝ : OrderedFinpartition n✝
        n : Nat
        c : OrderedFinpartition (HAdd.hAdd n 1)
        h : Eq (Set.range (c.emb 0)) (Singleton.singleton 0)
        A : Eq (HAdd.hAdd (HSub.hSub c.length 1) 1) c.length
        ⊢ Eq { length := HAdd.hAdd (HSub.hSub c.length 1) 1, partSize := Fin.cons 1 fu …
      -/
      ext
        /-
          case pos.length
          𝕜 : Type u_1
          inst✝⁶ : NontriviallyNormedField 𝕜
          E : Type u_2
          inst✝⁵ : NormedAddCommGroup E
          inst✝⁴ : NormedSpace 𝕜 E
          F : Type u_3
          inst✝³ : NormedAddCommGroup F
          inst✝² : NormedSpace 𝕜 F
          G : Type u_4
          inst✝¹ : NormedAddCommGroup G
          inst✝ : NormedSpace 𝕜 G
          s : Set E
          t : Set F
          q : F → FormalMultilinearSeries 𝕜 F G
          p : E → FormalMultilinearSeries 𝕜 E F
          n✝ : Nat
          c✝ : OrderedFinpartition n✝
          n : Nat
          c : OrderedFinpartition (HAdd.hAdd n 1)
          h : Eq (Set.range (c.emb 0)) (Singleton.singleton 0)
          A : Eq (HAdd.hAdd (HSub.hSub c.length 1) 1) c.length
          ⊢ Eq { length := HAdd.hAdd (HSub.hSub c.length 1) 1, partSize := Fin.cons 1 fu …
        -/
      · exact A
        /-
          🎉 no goals
        -/
        /-
          case pos.partSize
          𝕜 : Type u_1
          inst✝⁶ : NontriviallyNormedField 𝕜
          E : Type u_2
          inst✝⁵ : NormedAddCommGroup E
          inst✝⁴ : NormedSpace 𝕜 E
          F : Type u_3
          inst✝³ : NormedAddCommGroup F
          inst✝² : NormedSpace 𝕜 F
          G : Type u_4
          inst✝¹ : NormedAddCommGroup G
          inst✝ : NormedSpace 𝕜 G
          s : Set E
          t : Set F
          q : F → FormalMultilinearSeries 𝕜 F G
          p : E → FormalMultilinearSeries 𝕜 E F
          n✝ : Nat
          c✝ : OrderedFinpartition n✝
          n : Nat
          c : OrderedFinpartition (HAdd.hAdd n 1)
          h : Eq (Set.range (c.emb 0)) (Singleton.singleton 0)
          A : Eq (HAdd.hAdd (HSub.hSub c.length 1) 1) c.length
          ⊢ HEq { length := HAdd.hAdd (HSub.hSub c.length 1) 1, partSize := Fin.cons 1 f …
        -/
      · refine (Fin.heq_fun_iff A).mpr (fun i ↦ ?_)
        /-
          case pos.partSize
          𝕜 : Type u_1
          inst✝⁶ : NontriviallyNormedField 𝕜
          E : Type u_2
          inst✝⁵ : NormedAddCommGroup E
          inst✝⁴ : NormedSpace 𝕜 E
          F : Type u_3
          inst✝³ : NormedAddCommGroup F
          inst✝² : NormedSpace 𝕜 F
          G : Type u_4
          inst✝¹ : NormedAddCommGroup G
          inst✝ : NormedSpace 𝕜 G
          s : Set E
          t : Set F
          q : F → FormalMultilinearSeries 𝕜 F G
          p : E → FormalMultilinearSeries 𝕜 E F
          n✝ : Nat
          c✝ : OrderedFinpartition n✝
          n : Nat
          c : OrderedFinpartition (HAdd.hAdd n 1)
          h : Eq (Set.range (c.emb 0)) (Singleton.singleton 0)
          A : Eq (HAdd.hAdd (HSub.hSub c.length 1) 1) c.length
          i : Fin (HAdd.hAdd (HSub.hSub c.length 1) 1)
          ⊢ Eq ({ length := HAdd.hAdd (HSub.hSub c.length 1) 1, partSize := Fin.cons 1 f …
        -/
        simp [A]
        induction i using Fin.induction with
        | zero => change 1 = c.partSize 0; simp [c.partSize_eq_one_of_range_emb_eq_singleton h]
        | succ i => simp only [cons_succ, val_succ]; rfl
        /-
          case pos.emb
          𝕜 : Type u_1
          inst✝⁶ : NontriviallyNormedField 𝕜
          E : Type u_2
          inst✝⁵ : NormedAddCommGroup E
          inst✝⁴ : NormedSpace 𝕜 E
          F : Type u_3
          inst✝³ : NormedAddCommGroup F
          inst✝² : NormedSpace 𝕜 F
          G : Type u_4
          inst✝¹ : NormedAddCommGroup G
          inst✝ : NormedSpace 𝕜 G
          s : Set E
          t : Set F
          q : F → FormalMultilinearSeries 𝕜 F G
          p : E → FormalMultilinearSeries 𝕜 E F
          n✝ : Nat
          c✝ : OrderedFinpartition n✝
          n : Nat
          c : OrderedFinpartition (HAdd.hAdd n 1)
          h : Eq (Set.range (c.emb 0)) (Singleton.singleton 0)
          A : Eq (HAdd.hAdd (HSub.hSub c.length 1) 1) c.length
          ⊢ HEq { length := HAdd.hAdd (HSub.hSub c.length 1) 1, partSize := Fin.cons 1 f …
        -/
      · refine hfunext (congrArg Fin A) ?_
        /-
          case pos.emb
          𝕜 : Type u_1
          inst✝⁶ : NontriviallyNormedField 𝕜
          E : Type u_2
          inst✝⁵ : NormedAddCommGroup E
          inst✝⁴ : NormedSpace 𝕜 E
          F : Type u_3
          inst✝³ : NormedAddCommGroup F
          inst✝² : NormedSpace 𝕜 F
          G : Type u_4
          inst✝¹ : NormedAddCommGroup G
          inst✝ : NormedSpace 𝕜 G
          s : Set E
          t : Set F
          q : F → FormalMultilinearSeries 𝕜 F G
          p : E → FormalMultilinearSeries 𝕜 E F
          n✝ : Nat
          c✝ : OrderedFinpartition n✝
          n : Nat
          c : OrderedFinpartition (HAdd.hAdd n 1)
          h : Eq (Set.range (c.emb 0)) (Singleton.singleton 0)
          A : Eq (HAdd.hAdd (HSub.hSub c.length 1) 1) c.length
          ⊢ ∀ (a : Fin { length := HAdd.hAdd (HSub.hSub c.length 1) 1, partSize := Fin.c …
        -/
        simp only [id_eq]
        /-
          case pos.emb
          𝕜 : Type u_1
          inst✝⁶ : NontriviallyNormedField 𝕜
          E : Type u_2
          inst✝⁵ : NormedAddCommGroup E
          inst✝⁴ : NormedSpace 𝕜 E
          F : Type u_3
          inst✝³ : NormedAddCommGroup F
          inst✝² : NormedSpace 𝕜 F
          G : Type u_4
          inst✝¹ : NormedAddCommGroup G
          inst✝ : NormedSpace 𝕜 G
          s : Set E
          t : Set F
          q : F → FormalMultilinearSeries 𝕜 F G
          p : E → FormalMultilinearSeries 𝕜 E F
          n✝ : Nat
          c✝ : OrderedFinpartition n✝
          n : Nat
          c : OrderedFinpartition (HAdd.hAdd n 1)
          h : Eq (Set.range (c.emb 0)) (Singleton.singleton 0)
          A : Eq (HAdd.hAdd (HSub.hSub c.length 1) 1) c.length
          ⊢ ∀ (a : Fin (HAdd.hAdd (HSub.hSub c.length 1) 1)) (a' : Fin c.length), HEq a  …
        -/
        intro i i' h'
        /-
          case pos.emb
          𝕜 : Type u_1
          inst✝⁶ : NontriviallyNormedField 𝕜
          E : Type u_2
          inst✝⁵ : NormedAddCommGroup E
          inst✝⁴ : NormedSpace 𝕜 E
          F : Type u_3
          inst✝³ : NormedAddCommGroup F
          inst✝² : NormedSpace 𝕜 F
          G : Type u_4
          inst✝¹ : NormedAddCommGroup G
          inst✝ : NormedSpace 𝕜 G
          s : Set E
          t : Set F
          q : F → FormalMultilinearSeries 𝕜 F G
          p : E → FormalMultilinearSeries 𝕜 E F
          n✝ : Nat
          c✝ : OrderedFinpartition n✝
          n : Nat
          c : OrderedFinpartition (HAdd.hAdd n 1)
          h : Eq (Set.range (c.emb 0)) (Singleton.singleton 0)
          A : Eq (HAdd.hAdd (HSub.hSub c.length 1) 1) c.length
          i : Fin (HAdd.hAdd (HSub.hSub c.length 1) 1)
          i' : Fin c.length
          h' : HEq i i'
          ⊢ HEq (Fin.cases (motive := fun x => Fin (Fin.cons 1 (fun i => c.partSize (Fin …
        -/
        have : i' = Fin.cast A i := eq_of_val_eq (by apply val_eq_val_of_heq h'.symm)
        /-
          case pos.emb
          𝕜 : Type u_1
          inst✝⁶ : NontriviallyNormedField 𝕜
          E : Type u_2
          inst✝⁵ : NormedAddCommGroup E
          inst✝⁴ : NormedSpace 𝕜 E
          F : Type u_3
          inst✝³ : NormedAddCommGroup F
          inst✝² : NormedSpace 𝕜 F
          G : Type u_4
          inst✝¹ : NormedAddCommGroup G
          inst✝ : NormedSpace 𝕜 G
          s : Set E
          t : Set F
          q : F → FormalMultilinearSeries 𝕜 F G
          p : E → FormalMultilinearSeries 𝕜 E F
          n✝ : Nat
          c✝ : OrderedFinpartition n✝
          n : Nat
          c : OrderedFinpartition (HAdd.hAdd n 1)
          h : Eq (Set.range (c.emb 0)) (Singleton.singleton 0)
          A : Eq (HAdd.hAdd (HSub.hSub c.length 1) 1) c.length
          i : Fin (HAdd.hAdd (HSub.hSub c.length 1) 1)
          i' : Fin c.length
          h' : HEq i i'
          this : Eq i' (Fin.cast A i)
          ⊢ HEq (Fin.cases (motive := fun x => Fin (Fin.cons 1 (fun i => c.partSize (Fin …
        -/
        subst this
        /-
          case pos.emb
          𝕜 : Type u_1
          inst✝⁶ : NontriviallyNormedField 𝕜
          E : Type u_2
          inst✝⁵ : NormedAddCommGroup E
          inst✝⁴ : NormedSpace 𝕜 E
          F : Type u_3
          inst✝³ : NormedAddCommGroup F
          inst✝² : NormedSpace 𝕜 F
          G : Type u_4
          inst✝¹ : NormedAddCommGroup G
          inst✝ : NormedSpace 𝕜 G
          s : Set E
          t : Set F
          q : F → FormalMultilinearSeries 𝕜 F G
          p : E → FormalMultilinearSeries 𝕜 E F
          n✝ : Nat
          c✝ : OrderedFinpartition n✝
          n : Nat
          c : OrderedFinpartition (HAdd.hAdd n 1)
          h : Eq (Set.range (c.emb 0)) (Singleton.singleton 0)
          A : Eq (HAdd.hAdd (HSub.hSub c.length 1) 1) c.length
          i : Fin (HAdd.hAdd (HSub.hSub c.length 1) 1)
          h' : HEq i (Fin.cast A i)
          ⊢ HEq (Fin.cases (motive := fun x => Fin (Fin.cons 1 (fun i => c.partSize (Fin …
        -/
        refine (Fin.heq_fun_iff ?_).mpr ?_
        · induction i using Fin.induction with
          | zero => simp [c.partSize_eq_one_of_range_emb_eq_singleton h]
          | succ i => simp
          /-
            case pos.emb.refine_2
            𝕜 : Type u_1
            inst✝⁶ : NontriviallyNormedField 𝕜
            E : Type u_2
            inst✝⁵ : NormedAddCommGroup E
            inst✝⁴ : NormedSpace 𝕜 E
            F : Type u_3
            inst✝³ : NormedAddCommGroup F
            inst✝² : NormedSpace 𝕜 F
            G : Type u_4
            inst✝¹ : NormedAddCommGroup G
            inst✝ : NormedSpace 𝕜 G
            s : Set E
            t : Set F
            q : F → FormalMultilinearSeries 𝕜 F G
            p : E → FormalMultilinearSeries 𝕜 E F
            n✝ : Nat
            c✝ : OrderedFinpartition n✝
            n : Nat
            c : OrderedFinpartition (HAdd.hAdd n 1)
            h : Eq (Set.range (c.emb 0)) (Singleton.singleton 0)
            A : Eq (HAdd.hAdd (HSub.hSub c.length 1) 1) c.length
            i : Fin (HAdd.hAdd (HSub.hSub c.length 1) 1)
            h' : HEq i (Fin.cast A i)
            ⊢ ∀ (i_1 : Fin (Fin.cons 1 (fun i => c.partSize (Fin.cast ⋯ i.succ)) i)), Eq ( …
          -/
        · intro j
          induction i using Fin.induction with
          | zero =>
            simp only [cases_zero, cast_zero, val_eq_zero]
            exact (apply_eq_of_range_eq_singleton h _).symm
          | succ i => simp
      /-
        case neg
        𝕜 : Type u_1
        inst✝⁶ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝⁵ : NormedAddCommGroup E
        inst✝⁴ : NormedSpace 𝕜 E
        F : Type u_3
        inst✝³ : NormedAddCommGroup F
        inst✝² : NormedSpace 𝕜 F
        G : Type u_4
        inst✝¹ : NormedAddCommGroup G
        inst✝ : NormedSpace 𝕜 G
        s : Set E
        t : Set F
        q : F → FormalMultilinearSeries 𝕜 F G
        p : E → FormalMultilinearSeries 𝕜 E F
        n✝ : Nat
        c✝ : OrderedFinpartition n✝
        n : Nat
        c : OrderedFinpartition (HAdd.hAdd n 1)
        h : Not (Eq (Set.range (c.emb 0)) (Singleton.singleton 0))
        ⊢ Eq ((fun c => c.fst.extend c.snd) ((fun c => dite (Eq (Set.range (c.emb 0))  …
      -/
    · dsimp only
      /-
        case neg
        𝕜 : Type u_1
        inst✝⁶ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝⁵ : NormedAddCommGroup E
        inst✝⁴ : NormedSpace 𝕜 E
        F : Type u_3
        inst✝³ : NormedAddCommGroup F
        inst✝² : NormedSpace 𝕜 F
        G : Type u_4
        inst✝¹ : NormedAddCommGroup G
        inst✝ : NormedSpace 𝕜 G
        s : Set E
        t : Set F
        q : F → FormalMultilinearSeries 𝕜 F G
        p : E → FormalMultilinearSeries 𝕜 E F
        n✝ : Nat
        c✝ : OrderedFinpartition n✝
        n : Nat
        c : OrderedFinpartition (HAdd.hAdd n 1)
        h : Not (Eq (Set.range (c.emb 0)) (Singleton.singleton 0))
        ⊢ Eq ((dite (Eq (Set.range (c.emb 0)) (Singleton.singleton 0)) (fun h => ⟨c.er …
      -/
      rw [dif_neg h]
      have B : c.partSize (c.index 0) - 1 + 1 = c.partSize (c.index 0) :=
        Nat.sub_add_cancel (c.partSize_pos (c.index 0))
      /-
        case neg
        𝕜 : Type u_1
        inst✝⁶ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝⁵ : NormedAddCommGroup E
        inst✝⁴ : NormedSpace 𝕜 E
        F : Type u_3
        inst✝³ : NormedAddCommGroup F
        inst✝² : NormedSpace 𝕜 F
        G : Type u_4
        inst✝¹ : NormedAddCommGroup G
        inst✝ : NormedSpace 𝕜 G
        s : Set E
        t : Set F
        q : F → FormalMultilinearSeries 𝕜 F G
        p : E → FormalMultilinearSeries 𝕜 E F
        n✝ : Nat
        c✝ : OrderedFinpartition n✝
        n : Nat
        c : OrderedFinpartition (HAdd.hAdd n 1)
        h : Not (Eq (Set.range (c.emb 0)) (Singleton.singleton 0))
        B : Eq (HAdd.hAdd (HSub.hSub (c.partSize (c.index 0)) 1) 1) (c.partSize (c.ind …
        ⊢ Eq (⟨c.eraseMiddle h, Option.some (c.index 0)⟩.fst.extend ⟨c.eraseMiddle h,  …
      -/
      simp only [extend, extendMiddle, eraseMiddle, Nat.succ_eq_add_one, ↓reduceDIte]
      /-
        case neg
        𝕜 : Type u_1
        inst✝⁶ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝⁵ : NormedAddCommGroup E
        inst✝⁴ : NormedSpace 𝕜 E
        F : Type u_3
        inst✝³ : NormedAddCommGroup F
        inst✝² : NormedSpace 𝕜 F
        G : Type u_4
        inst✝¹ : NormedAddCommGroup G
        inst✝ : NormedSpace 𝕜 G
        s : Set E
        t : Set F
        q : F → FormalMultilinearSeries 𝕜 F G
        p : E → FormalMultilinearSeries 𝕜 E F
        n✝ : Nat
        c✝ : OrderedFinpartition n✝
        n : Nat
        c : OrderedFinpartition (HAdd.hAdd n 1)
        h : Not (Eq (Set.range (c.emb 0)) (Singleton.singleton 0))
        B : Eq (HAdd.hAdd (HSub.hSub (c.partSize (c.index 0)) 1) 1) (c.partSize (c.ind …
        ⊢ Eq { length := c.length, partSize := Function.update (Function.update c.part …
      -/
      ext
        /-
          case neg.length
          𝕜 : Type u_1
          inst✝⁶ : NontriviallyNormedField 𝕜
          E : Type u_2
          inst✝⁵ : NormedAddCommGroup E
          inst✝⁴ : NormedSpace 𝕜 E
          F : Type u_3
          inst✝³ : NormedAddCommGroup F
          inst✝² : NormedSpace 𝕜 F
          G : Type u_4
          inst✝¹ : NormedAddCommGroup G
          inst✝ : NormedSpace 𝕜 G
          s : Set E
          t : Set F
          q : F → FormalMultilinearSeries 𝕜 F G
          p : E → FormalMultilinearSeries 𝕜 E F
          n✝ : Nat
          c✝ : OrderedFinpartition n✝
          n : Nat
          c : OrderedFinpartition (HAdd.hAdd n 1)
          h : Not (Eq (Set.range (c.emb 0)) (Singleton.singleton 0))
          B : Eq (HAdd.hAdd (HSub.hSub (c.partSize (c.index 0)) 1) 1) (c.partSize (c.ind …
          ⊢ Eq { length := c.length, partSize := Function.update (Function.update c.part …
        -/
      · rfl
        /-
          🎉 no goals
        -/
        /-
          case neg.partSize
          𝕜 : Type u_1
          inst✝⁶ : NontriviallyNormedField 𝕜
          E : Type u_2
          inst✝⁵ : NormedAddCommGroup E
          inst✝⁴ : NormedSpace 𝕜 E
          F : Type u_3
          inst✝³ : NormedAddCommGroup F
          inst✝² : NormedSpace 𝕜 F
          G : Type u_4
          inst✝¹ : NormedAddCommGroup G
          inst✝ : NormedSpace 𝕜 G
          s : Set E
          t : Set F
          q : F → FormalMultilinearSeries 𝕜 F G
          p : E → FormalMultilinearSeries 𝕜 E F
          n✝ : Nat
          c✝ : OrderedFinpartition n✝
          n : Nat
          c : OrderedFinpartition (HAdd.hAdd n 1)
          h : Not (Eq (Set.range (c.emb 0)) (Singleton.singleton 0))
          B : Eq (HAdd.hAdd (HSub.hSub (c.partSize (c.index 0)) 1) 1) (c.partSize (c.ind …
          ⊢ HEq { length := c.length, partSize := Function.update (Function.update c.par …
        -/
      · simp only [update_self, update_idem, heq_eq_eq, update_eq_self_iff, B]
        /-
          🎉 no goals
        -/
        /-
          case neg.emb
          𝕜 : Type u_1
          inst✝⁶ : NontriviallyNormedField 𝕜
          E : Type u_2
          inst✝⁵ : NormedAddCommGroup E
          inst✝⁴ : NormedSpace 𝕜 E
          F : Type u_3
          inst✝³ : NormedAddCommGroup F
          inst✝² : NormedSpace 𝕜 F
          G : Type u_4
          inst✝¹ : NormedAddCommGroup G
          inst✝ : NormedSpace 𝕜 G
          s : Set E
          t : Set F
          q : F → FormalMultilinearSeries 𝕜 F G
          p : E → FormalMultilinearSeries 𝕜 E F
          n✝ : Nat
          c✝ : OrderedFinpartition n✝
          n : Nat
          c : OrderedFinpartition (HAdd.hAdd n 1)
          h : Not (Eq (Set.range (c.emb 0)) (Singleton.singleton 0))
          B : Eq (HAdd.hAdd (HSub.hSub (c.partSize (c.index 0)) 1) 1) (c.partSize (c.ind …
          ⊢ HEq { length := c.length, partSize := Function.update (Function.update c.par …
        -/
      · refine hfunext rfl ?_
        /-
          case neg.emb
          𝕜 : Type u_1
          inst✝⁶ : NontriviallyNormedField 𝕜
          E : Type u_2
          inst✝⁵ : NormedAddCommGroup E
          inst✝⁴ : NormedSpace 𝕜 E
          F : Type u_3
          inst✝³ : NormedAddCommGroup F
          inst✝² : NormedSpace 𝕜 F
          G : Type u_4
          inst✝¹ : NormedAddCommGroup G
          inst✝ : NormedSpace 𝕜 G
          s : Set E
          t : Set F
          q : F → FormalMultilinearSeries 𝕜 F G
          p : E → FormalMultilinearSeries 𝕜 E F
          n✝ : Nat
          c✝ : OrderedFinpartition n✝
          n : Nat
          c : OrderedFinpartition (HAdd.hAdd n 1)
          h : Not (Eq (Set.range (c.emb 0)) (Singleton.singleton 0))
          B : Eq (HAdd.hAdd (HSub.hSub (c.partSize (c.index 0)) 1) 1) (c.partSize (c.ind …
          ⊢ ∀ (a : Fin { length := c.length, partSize := Function.update (Function.updat …
        -/
        simp only [heq_eq_eq, forall_eq']
        /-
          case neg.emb
          𝕜 : Type u_1
          inst✝⁶ : NontriviallyNormedField 𝕜
          E : Type u_2
          inst✝⁵ : NormedAddCommGroup E
          inst✝⁴ : NormedSpace 𝕜 E
          F : Type u_3
          inst✝³ : NormedAddCommGroup F
          inst✝² : NormedSpace 𝕜 F
          G : Type u_4
          inst✝¹ : NormedAddCommGroup G
          inst✝ : NormedSpace 𝕜 G
          s : Set E
          t : Set F
          q : F → FormalMultilinearSeries 𝕜 F G
          p : E → FormalMultilinearSeries 𝕜 E F
          n✝ : Nat
          c✝ : OrderedFinpartition n✝
          n : Nat
          c : OrderedFinpartition (HAdd.hAdd n 1)
          h : Not (Eq (Set.range (c.emb 0)) (Singleton.singleton 0))
          B : Eq (HAdd.hAdd (HSub.hSub (c.partSize (c.index 0)) 1) 1) (c.partSize (c.ind …
          ⊢ ∀ (a : Fin c.length), HEq (dite (Eq a (c.index 0)) (fun h_1 => Function.comp …
        -/
        intro i
        /-
          case neg.emb
          𝕜 : Type u_1
          inst✝⁶ : NontriviallyNormedField 𝕜
          E : Type u_2
          inst✝⁵ : NormedAddCommGroup E
          inst✝⁴ : NormedSpace 𝕜 E
          F : Type u_3
          inst✝³ : NormedAddCommGroup F
          inst✝² : NormedSpace 𝕜 F
          G : Type u_4
          inst✝¹ : NormedAddCommGroup G
          inst✝ : NormedSpace 𝕜 G
          s : Set E
          t : Set F
          q : F → FormalMultilinearSeries 𝕜 F G
          p : E → FormalMultilinearSeries 𝕜 E F
          n✝ : Nat
          c✝ : OrderedFinpartition n✝
          n : Nat
          c : OrderedFinpartition (HAdd.hAdd n 1)
          h : Not (Eq (Set.range (c.emb 0)) (Singleton.singleton 0))
          B : Eq (HAdd.hAdd (HSub.hSub (c.partSize (c.index 0)) 1) 1) (c.partSize (c.ind …
          i : Fin c.length
          ⊢ HEq (dite (Eq i (c.index 0)) (fun h_1 => Function.comp (fun i => Fin.cases 0 …
        -/
        refine ((Fin.heq_fun_iff ?_).mpr ?_).symm
          /-
            case neg.emb.refine_1
            𝕜 : Type u_1
            inst✝⁶ : NontriviallyNormedField 𝕜
            E : Type u_2
            inst✝⁵ : NormedAddCommGroup E
            inst✝⁴ : NormedSpace 𝕜 E
            F : Type u_3
            inst✝³ : NormedAddCommGroup F
            inst✝² : NormedSpace 𝕜 F
            G : Type u_4
            inst✝¹ : NormedAddCommGroup G
            inst✝ : NormedSpace 𝕜 G
            s : Set E
            t : Set F
            q : F → FormalMultilinearSeries 𝕜 F G
            p : E → FormalMultilinearSeries 𝕜 E F
            n✝ : Nat
            c✝ : OrderedFinpartition n✝
            n : Nat
            c : OrderedFinpartition (HAdd.hAdd n 1)
            h : Not (Eq (Set.range (c.emb 0)) (Singleton.singleton 0))
            B : Eq (HAdd.hAdd (HSub.hSub (c.partSize (c.index 0)) 1) 1) (c.partSize (c.ind …
            i : Fin c.length
            ⊢ Eq (c.partSize i) (Function.update (Function.update c.partSize (c.index 0) ( …
          -/
        · simp only [update_self, B, update_idem, update_eq_self]
          /-
            🎉 no goals
          -/
          /-
            case neg.emb.refine_2
            𝕜 : Type u_1
            inst✝⁶ : NontriviallyNormedField 𝕜
            E : Type u_2
            inst✝⁵ : NormedAddCommGroup E
            inst✝⁴ : NormedSpace 𝕜 E
            F : Type u_3
            inst✝³ : NormedAddCommGroup F
            inst✝² : NormedSpace 𝕜 F
            G : Type u_4
            inst✝¹ : NormedAddCommGroup G
            inst✝ : NormedSpace 𝕜 G
            s : Set E
            t : Set F
            q : F → FormalMultilinearSeries 𝕜 F G
            p : E → FormalMultilinearSeries 𝕜 E F
            n✝ : Nat
            c✝ : OrderedFinpartition n✝
            n : Nat
            c : OrderedFinpartition (HAdd.hAdd n 1)
            h : Not (Eq (Set.range (c.emb 0)) (Singleton.singleton 0))
            B : Eq (HAdd.hAdd (HSub.hSub (c.partSize (c.index 0)) 1) 1) (c.partSize (c.ind …
            i : Fin c.length
            ⊢ ∀ (i_1 : Fin (c.partSize i)), Eq (c.emb i i_1) (dite (Eq i (c.index 0)) (fun …
          -/
        · intro j
          /-
            case neg.emb.refine_2
            𝕜 : Type u_1
            inst✝⁶ : NontriviallyNormedField 𝕜
            E : Type u_2
            inst✝⁵ : NormedAddCommGroup E
            inst✝⁴ : NormedSpace 𝕜 E
            F : Type u_3
            inst✝³ : NormedAddCommGroup F
            inst✝² : NormedSpace 𝕜 F
            G : Type u_4
            inst✝¹ : NormedAddCommGroup G
            inst✝ : NormedSpace 𝕜 G
            s : Set E
            t : Set F
            q : F → FormalMultilinearSeries 𝕜 F G
            p : E → FormalMultilinearSeries 𝕜 E F
            n✝ : Nat
            c✝ : OrderedFinpartition n✝
            n : Nat
            c : OrderedFinpartition (HAdd.hAdd n 1)
            h : Not (Eq (Set.range (c.emb 0)) (Singleton.singleton 0))
            B : Eq (HAdd.hAdd (HSub.hSub (c.partSize (c.index 0)) 1) 1) (c.partSize (c.ind …
            i : Fin c.length
            j : Fin (c.partSize i)
            ⊢ Eq (c.emb i j) (dite (Eq i (c.index 0)) (fun h_1 => Function.comp (fun i =>  …
          -/
          rcases eq_or_ne i (c.index 0) with rfl | hi
            /-
              case neg.emb.refine_2.inl
              𝕜 : Type u_1
              inst✝⁶ : NontriviallyNormedField 𝕜
              E : Type u_2
              inst✝⁵ : NormedAddCommGroup E
              inst✝⁴ : NormedSpace 𝕜 E
              F : Type u_3
              inst✝³ : NormedAddCommGroup F
              inst✝² : NormedSpace 𝕜 F
              G : Type u_4
              inst✝¹ : NormedAddCommGroup G
              inst✝ : NormedSpace 𝕜 G
              s : Set E
              t : Set F
              q : F → FormalMultilinearSeries 𝕜 F G
              p : E → FormalMultilinearSeries 𝕜 E F
              n✝ : Nat
              c✝ : OrderedFinpartition n✝
              n : Nat
              c : OrderedFinpartition (HAdd.hAdd n 1)
              h : Not (Eq (Set.range (c.emb 0)) (Singleton.singleton 0))
              B : Eq (HAdd.hAdd (HSub.hSub (c.partSize (c.index 0)) 1) 1) (c.partSize (c.ind …
              j : Fin (c.partSize (c.index 0))
              ⊢ Eq (c.emb (c.index 0) j) (dite (Eq (c.index 0) (c.index 0)) (fun h_1 => Func …
            -/
          · simp only [↓reduceDIte, comp_apply]
            /-
              case neg.emb.refine_2.inl
              𝕜 : Type u_1
              inst✝⁶ : NontriviallyNormedField 𝕜
              E : Type u_2
              inst✝⁵ : NormedAddCommGroup E
              inst✝⁴ : NormedSpace 𝕜 E
              F : Type u_3
              inst✝³ : NormedAddCommGroup F
              inst✝² : NormedSpace 𝕜 F
              G : Type u_4
              inst✝¹ : NormedAddCommGroup G
              inst✝ : NormedSpace 𝕜 G
              s : Set E
              t : Set F
              q : F → FormalMultilinearSeries 𝕜 F G
              p : E → FormalMultilinearSeries 𝕜 E F
              n✝ : Nat
              c✝ : OrderedFinpartition n✝
              n : Nat
              c : OrderedFinpartition (HAdd.hAdd n 1)
              h : Not (Eq (Set.range (c.emb 0)) (Singleton.singleton 0))
              B : Eq (HAdd.hAdd (HSub.hSub (c.partSize (c.index 0)) 1) 1) (c.partSize (c.ind …
              j : Fin (c.partSize (c.index 0))
              ⊢ Eq (c.emb (c.index 0) j) (Fin.cases 0 (Function.comp Fin.succ fun j => (c.em …
            -/
            rcases eq_or_ne j 0 with rfl | hj
              /-
                case neg.emb.refine_2.inl.inl
                𝕜 : Type u_1
                inst✝⁶ : NontriviallyNormedField 𝕜
                E : Type u_2
                inst✝⁵ : NormedAddCommGroup E
                inst✝⁴ : NormedSpace 𝕜 E
                F : Type u_3
                inst✝³ : NormedAddCommGroup F
                inst✝² : NormedSpace 𝕜 F
                G : Type u_4
                inst✝¹ : NormedAddCommGroup G
                inst✝ : NormedSpace 𝕜 G
                s : Set E
                t : Set F
                q : F → FormalMultilinearSeries 𝕜 F G
                p : E → FormalMultilinearSeries 𝕜 E F
                n✝ : Nat
                c✝ : OrderedFinpartition n✝
                n : Nat
                c : OrderedFinpartition (HAdd.hAdd n 1)
                h : Not (Eq (Set.range (c.emb 0)) (Singleton.singleton 0))
                B : Eq (HAdd.hAdd (HSub.hSub (c.partSize (c.index 0)) 1) 1) (c.partSize (c.ind …
                ⊢ Eq (c.emb (c.index 0) 0) (Fin.cases 0 (Function.comp Fin.succ fun j => (c.em …
              -/
            · simpa using c.emb_zero
              /-
                🎉 no goals
              -/
              /-
                case neg.emb.refine_2.inl.inr
                𝕜 : Type u_1
                inst✝⁶ : NontriviallyNormedField 𝕜
                E : Type u_2
                inst✝⁵ : NormedAddCommGroup E
                inst✝⁴ : NormedSpace 𝕜 E
                F : Type u_3
                inst✝³ : NormedAddCommGroup F
                inst✝² : NormedSpace 𝕜 F
                G : Type u_4
                inst✝¹ : NormedAddCommGroup G
                inst✝ : NormedSpace 𝕜 G
                s : Set E
                t : Set F
                q : F → FormalMultilinearSeries 𝕜 F G
                p : E → FormalMultilinearSeries 𝕜 E F
                n✝ : Nat
                c✝ : OrderedFinpartition n✝
                n : Nat
                c : OrderedFinpartition (HAdd.hAdd n 1)
                h : Not (Eq (Set.range (c.emb 0)) (Singleton.singleton 0))
                B : Eq (HAdd.hAdd (HSub.hSub (c.partSize (c.index 0)) 1) 1) (c.partSize (c.ind …
                j : Fin (c.partSize (c.index 0))
                hj : Ne j 0
                ⊢ Eq (c.emb (c.index 0) j) (Fin.cases 0 (Function.comp Fin.succ fun j => (c.em …
              -/
            · let j' := Fin.pred (cast B.symm j) (by simpa using hj)
              /-
                case neg.emb.refine_2.inl.inr
                𝕜 : Type u_1
                inst✝⁶ : NontriviallyNormedField 𝕜
                E : Type u_2
                inst✝⁵ : NormedAddCommGroup E
                inst✝⁴ : NormedSpace 𝕜 E
                F : Type u_3
                inst✝³ : NormedAddCommGroup F
                inst✝² : NormedSpace 𝕜 F
                G : Type u_4
                inst✝¹ : NormedAddCommGroup G
                inst✝ : NormedSpace 𝕜 G
                s : Set E
                t : Set F
                q : F → FormalMultilinearSeries 𝕜 F G
                p : E → FormalMultilinearSeries 𝕜 E F
                n✝ : Nat
                c✝ : OrderedFinpartition n✝
                n : Nat
                c : OrderedFinpartition (HAdd.hAdd n 1)
                h : Not (Eq (Set.range (c.emb 0)) (Singleton.singleton 0))
                B : Eq (HAdd.hAdd (HSub.hSub (c.partSize (c.index 0)) 1) 1) (c.partSize (c.ind …
                j : Fin (c.partSize (c.index 0))
                hj : Ne j 0
                j' : Fin (HSub.hSub (c.partSize (c.index 0)) 1) := (Fin.cast ⋯ j).pred ⋯
                ⊢ Eq (c.emb (c.index 0) j) (Fin.cases 0 (Function.comp Fin.succ fun j => (c.em …
              -/
              have : j = cast B (succ j') := by simp [j']
              simp only [this, coe_cast, val_succ, cast_mk, cases_succ', comp_apply, succ_mk,
                Nat.succ_eq_add_one, succ_pred]
              /-
                case neg.emb.refine_2.inl.inr
                𝕜 : Type u_1
                inst✝⁶ : NontriviallyNormedField 𝕜
                E : Type u_2
                inst✝⁵ : NormedAddCommGroup E
                inst✝⁴ : NormedSpace 𝕜 E
                F : Type u_3
                inst✝³ : NormedAddCommGroup F
                inst✝² : NormedSpace 𝕜 F
                G : Type u_4
                inst✝¹ : NormedAddCommGroup G
                inst✝ : NormedSpace 𝕜 G
                s : Set E
                t : Set F
                q : F → FormalMultilinearSeries 𝕜 F G
                p : E → FormalMultilinearSeries 𝕜 E F
                n✝ : Nat
                c✝ : OrderedFinpartition n✝
                n : Nat
                c : OrderedFinpartition (HAdd.hAdd n 1)
                h : Not (Eq (Set.range (c.emb 0)) (Singleton.singleton 0))
                B : Eq (HAdd.hAdd (HSub.hSub (c.partSize (c.index 0)) 1) 1) (c.partSize (c.ind …
                j : Fin (c.partSize (c.index 0))
                hj : Ne j 0
                j' : Fin (HSub.hSub (c.partSize (c.index 0)) 1) := (Fin.cast ⋯ j).pred ⋯
                this : Eq j (Fin.cast B j'.succ)
                ⊢ Eq (c.emb (c.index 0) (Fin.cast B j'.succ)) (c.emb (c.index 0) ⟨HAdd.hAdd (↑ …
              -/
              rfl
              /-
                🎉 no goals
              -/
            /-
              case neg.emb.refine_2.inr
              𝕜 : Type u_1
              inst✝⁶ : NontriviallyNormedField 𝕜
              E : Type u_2
              inst✝⁵ : NormedAddCommGroup E
              inst✝⁴ : NormedSpace 𝕜 E
              F : Type u_3
              inst✝³ : NormedAddCommGroup F
              inst✝² : NormedSpace 𝕜 F
              G : Type u_4
              inst✝¹ : NormedAddCommGroup G
              inst✝ : NormedSpace 𝕜 G
              s : Set E
              t : Set F
              q : F → FormalMultilinearSeries 𝕜 F G
              p : E → FormalMultilinearSeries 𝕜 E F
              n✝ : Nat
              c✝ : OrderedFinpartition n✝
              n : Nat
              c : OrderedFinpartition (HAdd.hAdd n 1)
              h : Not (Eq (Set.range (c.emb 0)) (Singleton.singleton 0))
              B : Eq (HAdd.hAdd (HSub.hSub (c.partSize (c.index 0)) 1) 1) (c.partSize (c.ind …
              i : Fin c.length
              j : Fin (c.partSize i)
              hi : Ne i (c.index 0)
              ⊢ Eq (c.emb i j) (dite (Eq i (c.index 0)) (fun h_1 => Function.comp (fun i =>  …
            -/
          · simp [hi]
            /-
              🎉 no goals
            -/


/-- Given a formal multilinear series `p`, an ordered partition `c` of `n` and the index `i` of a
block of `c`, we may define a function on `Fin n → E` by picking the variables in the `i`-th block
of `n`, and applying the corresponding coefficient of `p` to these variables. This function is
called `p.applyOrderedFinpartition c v i` for `v : Fin n → E` and `i : Fin c.k`. -/
def applyOrderedFinpartition (p : ∀ (i : Fin c.length), E[×c.partSize i]→L[𝕜] F) :
    (Fin n → E) → Fin c.length → F :=
  fun v m ↦ p m (v ∘ c.emb m)


lemma applyOrderedFinpartition_apply (p : ∀ (i : Fin c.length), E[×c.partSize i]→L[𝕜] F)
    (v : Fin n → E) :
  c.applyOrderedFinpartition p v = (fun m ↦ p m (v ∘ c.emb m)) := rfl


theorem norm_applyOrderedFinpartition_le (p : ∀ (i : Fin c.length), E[×c.partSize i]→L[𝕜] F)
    (v : Fin n → E) (m : Fin c.length) :
    ‖c.applyOrderedFinpartition p v m‖ ≤ ‖p m‖ * ∏ i : Fin (c.partSize m), ‖v (c.emb m i)‖ :=
  (p m).le_opNorm _


/-- Technical lemma stating how `c.applyOrderedFinpartition` commutes with updating variables. This
will be the key point to show that functions constructed from `applyOrderedFinpartition` retain
multilinearity. -/
theorem applyOrderedFinpartition_update_right
    (p : ∀ (i : Fin c.length), E[×c.partSize i]→L[𝕜] F)
    (j : Fin n) (v : Fin n → E) (z : E) :
    c.applyOrderedFinpartition p (update v j z) =
      update (c.applyOrderedFinpartition p v) (c.index j)
        (p (c.index j)
          (Function.update (v ∘ c.emb (c.index j)) (c.invEmbedding j) z)) := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    n : Nat
    c : OrderedFinpartition n
    p : (i : Fin c.length) → ContinuousMultilinearMap 𝕜 (fun i => E) F
    j : Fin n
    v : Fin n → E
    z : E
    ⊢ Eq (c.applyOrderedFinpartition p (Function.update v j z)) (Function.update ( …
  -/
  ext m
  /-
    case h
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    n : Nat
    c : OrderedFinpartition n
    p : (i : Fin c.length) → ContinuousMultilinearMap 𝕜 (fun i => E) F
    j : Fin n
    v : Fin n → E
    z : E
    m : Fin c.length
    ⊢ Eq (c.applyOrderedFinpartition p (Function.update v j z) m) (Function.update …
  -/
  by_cases h : m = c.index j
    /-
      case pos
      𝕜 : Type u_1
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      F : Type u_3
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      n : Nat
      c : OrderedFinpartition n
      p : (i : Fin c.length) → ContinuousMultilinearMap 𝕜 (fun i => E) F
      j : Fin n
      v : Fin n → E
      z : E
      m : Fin c.length
      h : Eq m (c.index j)
      ⊢ Eq (c.applyOrderedFinpartition p (Function.update v j z) m) (Function.update …
    -/
  · rw [h]
    /-
      case pos
      𝕜 : Type u_1
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      F : Type u_3
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      n : Nat
      c : OrderedFinpartition n
      p : (i : Fin c.length) → ContinuousMultilinearMap 𝕜 (fun i => E) F
      j : Fin n
      v : Fin n → E
      z : E
      m : Fin c.length
      h : Eq m (c.index j)
      ⊢ Eq (c.applyOrderedFinpartition p (Function.update v j z) (c.index j)) (Funct …
    -/
    simp only [applyOrderedFinpartition, update_self]
    /-
      case pos
      𝕜 : Type u_1
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      F : Type u_3
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      n : Nat
      c : OrderedFinpartition n
      p : (i : Fin c.length) → ContinuousMultilinearMap 𝕜 (fun i => E) F
      j : Fin n
      v : Fin n → E
      z : E
      m : Fin c.length
      h : Eq m (c.index j)
      ⊢ Eq ((p (c.index j)) (Function.comp (Function.update v j z) (c.emb (c.index j …
    -/
    congr
    /-
      case pos.h.e_6.h
      𝕜 : Type u_1
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      F : Type u_3
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      n : Nat
      c : OrderedFinpartition n
      p : (i : Fin c.length) → ContinuousMultilinearMap 𝕜 (fun i => E) F
      j : Fin n
      v : Fin n → E
      z : E
      m : Fin c.length
      h : Eq m (c.index j)
      ⊢ Eq (Function.comp (Function.update v j z) (c.emb (c.index j))) (Function.upd …
    -/
    rw [← Function.update_comp_eq_of_injective]
      /-
        case pos.h.e_6.h
        𝕜 : Type u_1
        inst✝⁴ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝³ : NormedAddCommGroup E
        inst✝² : NormedSpace 𝕜 E
        F : Type u_3
        inst✝¹ : NormedAddCommGroup F
        inst✝ : NormedSpace 𝕜 F
        n : Nat
        c : OrderedFinpartition n
        p : (i : Fin c.length) → ContinuousMultilinearMap 𝕜 (fun i => E) F
        j : Fin n
        v : Fin n → E
        z : E
        m : Fin c.length
        h : Eq m (c.index j)
        ⊢ Eq (Function.comp (Function.update v j z) (c.emb (c.index j))) (Function.com …
      -/
    · simp
      /-
        🎉 no goals
      -/
      /-
        case pos.h.e_6.h.hf
        𝕜 : Type u_1
        inst✝⁴ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝³ : NormedAddCommGroup E
        inst✝² : NormedSpace 𝕜 E
        F : Type u_3
        inst✝¹ : NormedAddCommGroup F
        inst✝ : NormedSpace 𝕜 F
        n : Nat
        c : OrderedFinpartition n
        p : (i : Fin c.length) → ContinuousMultilinearMap 𝕜 (fun i => E) F
        j : Fin n
        v : Fin n → E
        z : E
        m : Fin c.length
        h : Eq m (c.index j)
        ⊢ Function.Injective (c.emb (c.index j))
      -/
    · exact (c.emb_strictMono (c.index j)).injective
      /-
        🎉 no goals
      -/
  · simp only [applyOrderedFinpartition, ne_eq, h, not_false_eq_true,
      update_of_ne]
    /-
      case neg
      𝕜 : Type u_1
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      F : Type u_3
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      n : Nat
      c : OrderedFinpartition n
      p : (i : Fin c.length) → ContinuousMultilinearMap 𝕜 (fun i => E) F
      j : Fin n
      v : Fin n → E
      z : E
      m : Fin c.length
      h : Not (Eq m (c.index j))
      ⊢ Eq ((p m) (Function.comp (Function.update v j z) (c.emb m))) ((p m) (Functio …
    -/
    congr
    /-
      case neg.h.e_6.h
      𝕜 : Type u_1
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      F : Type u_3
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      n : Nat
      c : OrderedFinpartition n
      p : (i : Fin c.length) → ContinuousMultilinearMap 𝕜 (fun i => E) F
      j : Fin n
      v : Fin n → E
      z : E
      m : Fin c.length
      h : Not (Eq m (c.index j))
      ⊢ Eq (Function.comp (Function.update v j z) (c.emb m)) (Function.comp v (c.emb …
    -/
    apply Function.update_comp_eq_of_not_mem_range
    have A : Disjoint (range (c.emb m)) (range (c.emb (c.index j))) :=
      c.disjoint (mem_univ m) (mem_univ (c.index j)) h
    /-
      case neg.h.e_6.h.h
      𝕜 : Type u_1
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      F : Type u_3
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      n : Nat
      c : OrderedFinpartition n
      p : (i : Fin c.length) → ContinuousMultilinearMap 𝕜 (fun i => E) F
      j : Fin n
      v : Fin n → E
      z : E
      m : Fin c.length
      h : Not (Eq m (c.index j))
      A : Disjoint (Set.range (c.emb m)) (Set.range (c.emb (c.index j)))
      ⊢ Not (Membership.mem (Set.range (c.emb m)) j)
    -/
    have : j ∈ range (c.emb (c.index j)) := mem_range.2 ⟨c.invEmbedding j, by simp⟩
    /-
      case neg.h.e_6.h.h
      𝕜 : Type u_1
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      F : Type u_3
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      n : Nat
      c : OrderedFinpartition n
      p : (i : Fin c.length) → ContinuousMultilinearMap 𝕜 (fun i => E) F
      j : Fin n
      v : Fin n → E
      z : E
      m : Fin c.length
      h : Not (Eq m (c.index j))
      A : Disjoint (Set.range (c.emb m)) (Set.range (c.emb (c.index j)))
      this : Membership.mem (Set.range (c.emb (c.index j))) j
      ⊢ Not (Membership.mem (Set.range (c.emb m)) j)
    -/
    exact Set.disjoint_right.1 A this
    /-
      🎉 no goals
    -/


theorem applyOrderedFinpartition_update_left (p : ∀ (i : Fin c.length), E[×c.partSize i]→L[𝕜] F)
    (m : Fin c.length) (v : Fin n → E) (q : E[×c.partSize m]→L[𝕜] F) :
    c.applyOrderedFinpartition (update p m q) v
      = update (c.applyOrderedFinpartition p v) m (q (v ∘ c.emb m)) := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    n : Nat
    c : OrderedFinpartition n
    p : (i : Fin c.length) → ContinuousMultilinearMap 𝕜 (fun i => E) F
    m : Fin c.length
    v : Fin n → E
    q : ContinuousMultilinearMap 𝕜 (fun i => E) F
    ⊢ Eq (c.applyOrderedFinpartition (Function.update p m q) v) (Function.update ( …
  -/
  ext d
  /-
    case h
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    n : Nat
    c : OrderedFinpartition n
    p : (i : Fin c.length) → ContinuousMultilinearMap 𝕜 (fun i => E) F
    m : Fin c.length
    v : Fin n → E
    q : ContinuousMultilinearMap 𝕜 (fun i => E) F
    d : Fin c.length
    ⊢ Eq (c.applyOrderedFinpartition (Function.update p m q) v d) (Function.update …
  -/
  by_cases h : d = m
    /-
      case pos
      𝕜 : Type u_1
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      F : Type u_3
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      n : Nat
      c : OrderedFinpartition n
      p : (i : Fin c.length) → ContinuousMultilinearMap 𝕜 (fun i => E) F
      m : Fin c.length
      v : Fin n → E
      q : ContinuousMultilinearMap 𝕜 (fun i => E) F
      d : Fin c.length
      h : Eq d m
      ⊢ Eq (c.applyOrderedFinpartition (Function.update p m q) v d) (Function.update …
    -/
  · rw [h]
    /-
      case pos
      𝕜 : Type u_1
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      F : Type u_3
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      n : Nat
      c : OrderedFinpartition n
      p : (i : Fin c.length) → ContinuousMultilinearMap 𝕜 (fun i => E) F
      m : Fin c.length
      v : Fin n → E
      q : ContinuousMultilinearMap 𝕜 (fun i => E) F
      d : Fin c.length
      h : Eq d m
      ⊢ Eq (c.applyOrderedFinpartition (Function.update p m q) v m) (Function.update …
    -/
    simp [applyOrderedFinpartition]
    /-
      🎉 no goals
    -/
    /-
      case neg
      𝕜 : Type u_1
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      F : Type u_3
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      n : Nat
      c : OrderedFinpartition n
      p : (i : Fin c.length) → ContinuousMultilinearMap 𝕜 (fun i => E) F
      m : Fin c.length
      v : Fin n → E
      q : ContinuousMultilinearMap 𝕜 (fun i => E) F
      d : Fin c.length
      h : Not (Eq d m)
      ⊢ Eq (c.applyOrderedFinpartition (Function.update p m q) v d) (Function.update …
    -/
  · simp [h, applyOrderedFinpartition]
    /-
      🎉 no goals
    -/


/-- Given a an ordered finite partition `c` of `n`, a continuous multilinear map `f` in `c.length`
variables, and for each `m` a continuous multilinear map `p m` in `c.partSize m` variables,
one can form a continuous multilinear map in `n`
variables by applying `p m` to each part of the partition, and then
applying `f` to the resulting vector. It is called `c.compAlongOrderedFinpartition f p`. -/
def compAlongOrderedFinpartition (f : F [×c.length]→L[𝕜] G) (p : ∀ i, E [×c.partSize i]→L[𝕜] F) :
    E[×n]→L[𝕜] G where
  toFun v := f (c.applyOrderedFinpartition p v)
  map_update_add' v i x y := by
    /-
      𝕜 : Type u_1
      inst✝⁷ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁶ : NormedAddCommGroup E
      inst✝⁵ : NormedSpace 𝕜 E
      F : Type u_3
      inst✝⁴ : NormedAddCommGroup F
      inst✝³ : NormedSpace 𝕜 F
      G : Type u_4
      inst✝² : NormedAddCommGroup G
      inst✝¹ : NormedSpace 𝕜 G
      s : Set E
      t : Set F
      q : F → FormalMultilinearSeries 𝕜 F G
      p✝ : E → FormalMultilinearSeries 𝕜 E F
      n : Nat
      c : OrderedFinpartition n
      f : ContinuousMultilinearMap 𝕜 (fun i => F) G
      p : (i : Fin c.length) → ContinuousMultilinearMap 𝕜 (fun i => E) F
      inst✝ : DecidableEq (Fin n)
      v : Fin n → E
      i : Fin n
      x y : E
      ⊢ Eq ((fun v => f (c.applyOrderedFinpartition p v)) (Function.update v i (HAdd …
    -/
    cases Subsingleton.elim ‹_› (instDecidableEqFin _)
    /-
      case refl
      𝕜 : Type u_1
      inst✝⁶ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace 𝕜 E
      F : Type u_3
      inst✝³ : NormedAddCommGroup F
      inst✝² : NormedSpace 𝕜 F
      G : Type u_4
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace 𝕜 G
      s : Set E
      t : Set F
      q : F → FormalMultilinearSeries 𝕜 F G
      p✝ : E → FormalMultilinearSeries 𝕜 E F
      n : Nat
      c : OrderedFinpartition n
      f : ContinuousMultilinearMap 𝕜 (fun i => F) G
      p : (i : Fin c.length) → ContinuousMultilinearMap 𝕜 (fun i => E) F
      v : Fin n → E
      i : Fin n
      x y : E
      ⊢ Eq ((fun v => f (c.applyOrderedFinpartition p v)) (Function.update v i (HAdd …
    -/
    simp only [applyOrderedFinpartition_update_right, ContinuousMultilinearMap.map_update_add]
    /-
      🎉 no goals
    -/
  map_update_smul' v i c x := by
    /-
      𝕜 : Type u_1
      inst✝⁷ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁶ : NormedAddCommGroup E
      inst✝⁵ : NormedSpace 𝕜 E
      F : Type u_3
      inst✝⁴ : NormedAddCommGroup F
      inst✝³ : NormedSpace 𝕜 F
      G : Type u_4
      inst✝² : NormedAddCommGroup G
      inst✝¹ : NormedSpace 𝕜 G
      s : Set E
      t : Set F
      q : F → FormalMultilinearSeries 𝕜 F G
      p✝ : E → FormalMultilinearSeries 𝕜 E F
      n : Nat
      c✝ : OrderedFinpartition n
      f : ContinuousMultilinearMap 𝕜 (fun i => F) G
      p : (i : Fin c✝.length) → ContinuousMultilinearMap 𝕜 (fun i => E) F
      inst✝ : DecidableEq (Fin n)
      v : Fin n → E
      i : Fin n
      c : 𝕜
      x : E
      ⊢ Eq ((fun v => f (c✝.applyOrderedFinpartition p v)) (Function.update v i (HSM …
    -/
    cases Subsingleton.elim ‹_› (instDecidableEqFin _)
    /-
      case refl
      𝕜 : Type u_1
      inst✝⁶ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace 𝕜 E
      F : Type u_3
      inst✝³ : NormedAddCommGroup F
      inst✝² : NormedSpace 𝕜 F
      G : Type u_4
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace 𝕜 G
      s : Set E
      t : Set F
      q : F → FormalMultilinearSeries 𝕜 F G
      p✝ : E → FormalMultilinearSeries 𝕜 E F
      n : Nat
      c✝ : OrderedFinpartition n
      f : ContinuousMultilinearMap 𝕜 (fun i => F) G
      p : (i : Fin c✝.length) → ContinuousMultilinearMap 𝕜 (fun i => E) F
      v : Fin n → E
      i : Fin n
      c : 𝕜
      x : E
      ⊢ Eq ((fun v => f (c✝.applyOrderedFinpartition p v)) (Function.update v i (HSM …
    -/
    simp only [applyOrderedFinpartition_update_right, ContinuousMultilinearMap.map_update_smul]
    /-
      🎉 no goals
    -/
  cont := by
    /-
      𝕜 : Type u_1
      inst✝⁶ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace 𝕜 E
      F : Type u_3
      inst✝³ : NormedAddCommGroup F
      inst✝² : NormedSpace 𝕜 F
      G : Type u_4
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace 𝕜 G
      s : Set E
      t : Set F
      q : F → FormalMultilinearSeries 𝕜 F G
      p✝ : E → FormalMultilinearSeries 𝕜 E F
      n : Nat
      c : OrderedFinpartition n
      f : ContinuousMultilinearMap 𝕜 (fun i => F) G
      p : (i : Fin c.length) → ContinuousMultilinearMap 𝕜 (fun i => E) F
      ⊢ Continuous { toFun := fun v => f (c.applyOrderedFinpartition p v), map_updat …
    -/
    apply f.cont.comp
    /-
      𝕜 : Type u_1
      inst✝⁶ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace 𝕜 E
      F : Type u_3
      inst✝³ : NormedAddCommGroup F
      inst✝² : NormedSpace 𝕜 F
      G : Type u_4
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace 𝕜 G
      s : Set E
      t : Set F
      q : F → FormalMultilinearSeries 𝕜 F G
      p✝ : E → FormalMultilinearSeries 𝕜 E F
      n : Nat
      c : OrderedFinpartition n
      f : ContinuousMultilinearMap 𝕜 (fun i => F) G
      p : (i : Fin c.length) → ContinuousMultilinearMap 𝕜 (fun i => E) F
      ⊢ Continuous (c.applyOrderedFinpartition p)
    -/
    change Continuous (fun v m ↦ p m (v ∘ c.emb m))
    /-
      𝕜 : Type u_1
      inst✝⁶ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace 𝕜 E
      F : Type u_3
      inst✝³ : NormedAddCommGroup F
      inst✝² : NormedSpace 𝕜 F
      G : Type u_4
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace 𝕜 G
      s : Set E
      t : Set F
      q : F → FormalMultilinearSeries 𝕜 F G
      p✝ : E → FormalMultilinearSeries 𝕜 E F
      n : Nat
      c : OrderedFinpartition n
      f : ContinuousMultilinearMap 𝕜 (fun i => F) G
      p : (i : Fin c.length) → ContinuousMultilinearMap 𝕜 (fun i => E) F
      ⊢ Continuous fun v m => (p m) (Function.comp v (c.emb m))
    -/
    fun_prop
    /-
      🎉 no goals
    -/


@[simp] lemma compAlongOrderFinpartition_apply (f : F [×c.length]→L[𝕜] G)
    (p : ∀ i, E[×c.partSize i]→L[𝕜] F) (v : Fin n → E) :
    c.compAlongOrderedFinpartition f p v = f (c.applyOrderedFinpartition p v) := rfl


theorem norm_compAlongOrderedFinpartition_le (f : F [×c.length]→L[𝕜] G)
    (p : ∀ i, E [×c.partSize i]→L[𝕜] F) :
    ‖c.compAlongOrderedFinpartition f p‖ ≤ ‖f‖ * ∏ i, ‖p i‖ := by
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    G : Type u_4
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    n : Nat
    c : OrderedFinpartition n
    f : ContinuousMultilinearMap 𝕜 (fun i => F) G
    p : (i : Fin c.length) → ContinuousMultilinearMap 𝕜 (fun i => E) F
    ⊢ LE.le (Norm.norm (c.compAlongOrderedFinpartition f p)) (HMul.hMul (Norm.norm …
  -/
  refine ContinuousMultilinearMap.opNorm_le_bound (by positivity) fun v ↦ ?_
  rw [compAlongOrderFinpartition_apply, mul_assoc, ← c.prod_sigma_eq_prod,
    ← Finset.prod_mul_distrib]
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    G : Type u_4
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    n : Nat
    c : OrderedFinpartition n
    f : ContinuousMultilinearMap 𝕜 (fun i => F) G
    p : (i : Fin c.length) → ContinuousMultilinearMap 𝕜 (fun i => E) F
    v : Fin n → E
    ⊢ LE.le (Norm.norm (f (c.applyOrderedFinpartition p v))) (HMul.hMul (Norm.norm …
  -/
  exact f.le_opNorm_mul_prod_of_le <| c.norm_applyOrderedFinpartition_le _ _
  /-
    🎉 no goals
  -/


/-- Bundled version of `compAlongOrderedFinpartition`, depending linearly on `f`
and multilinearly on `p`.-/
@[simps apply_apply]
def compAlongOrderedFinpartitionₗ :
    (F [×c.length]→L[𝕜] G) →ₗ[𝕜]
      MultilinearMap 𝕜 (fun i : Fin c.length ↦ E[×c.partSize i]→L[𝕜] F) (E[×n]→L[𝕜] G) where
  toFun f :=
    { toFun := fun p ↦ c.compAlongOrderedFinpartition f p
      map_update_add' := by
        /-
          𝕜 : Type u_1
          inst✝⁶ : NontriviallyNormedField 𝕜
          E : Type u_2
          inst✝⁵ : NormedAddCommGroup E
          inst✝⁴ : NormedSpace 𝕜 E
          F : Type u_3
          inst✝³ : NormedAddCommGroup F
          inst✝² : NormedSpace 𝕜 F
          G : Type u_4
          inst✝¹ : NormedAddCommGroup G
          inst✝ : NormedSpace 𝕜 G
          s : Set E
          t : Set F
          q : F → FormalMultilinearSeries 𝕜 F G
          p : E → FormalMultilinearSeries 𝕜 E F
          n : Nat
          c : OrderedFinpartition n
          f : ContinuousMultilinearMap 𝕜 (fun i => F) G
          ⊢ ∀ [inst : DecidableEq (Fin c.length)] (m : (i : Fin c.length) → ContinuousMu …
        -/
        intro inst p m q q'
        /-
          𝕜 : Type u_1
          inst✝⁶ : NontriviallyNormedField 𝕜
          E : Type u_2
          inst✝⁵ : NormedAddCommGroup E
          inst✝⁴ : NormedSpace 𝕜 E
          F : Type u_3
          inst✝³ : NormedAddCommGroup F
          inst✝² : NormedSpace 𝕜 F
          G : Type u_4
          inst✝¹ : NormedAddCommGroup G
          inst✝ : NormedSpace 𝕜 G
          s : Set E
          t : Set F
          q✝ : F → FormalMultilinearSeries 𝕜 F G
          p✝ : E → FormalMultilinearSeries 𝕜 E F
          n : Nat
          c : OrderedFinpartition n
          f : ContinuousMultilinearMap 𝕜 (fun i => F) G
          inst : DecidableEq (Fin c.length)
          p : (i : Fin c.length) → ContinuousMultilinearMap 𝕜 (fun i => E) F
          m : Fin c.length
          q q' : ContinuousMultilinearMap 𝕜 (fun i => E) F
          ⊢ Eq ((fun p => c.compAlongOrderedFinpartition f p) (Function.update p m (HAdd …
        -/
        cases Subsingleton.elim ‹_› (instDecidableEqFin _)
        /-
          case refl
          𝕜 : Type u_1
          inst✝⁶ : NontriviallyNormedField 𝕜
          E : Type u_2
          inst✝⁵ : NormedAddCommGroup E
          inst✝⁴ : NormedSpace 𝕜 E
          F : Type u_3
          inst✝³ : NormedAddCommGroup F
          inst✝² : NormedSpace 𝕜 F
          G : Type u_4
          inst✝¹ : NormedAddCommGroup G
          inst✝ : NormedSpace 𝕜 G
          s : Set E
          t : Set F
          q✝ : F → FormalMultilinearSeries 𝕜 F G
          p✝ : E → FormalMultilinearSeries 𝕜 E F
          n : Nat
          c : OrderedFinpartition n
          f : ContinuousMultilinearMap 𝕜 (fun i => F) G
          p : (i : Fin c.length) → ContinuousMultilinearMap 𝕜 (fun i => E) F
          m : Fin c.length
          q q' : ContinuousMultilinearMap 𝕜 (fun i => E) F
          ⊢ Eq ((fun p => c.compAlongOrderedFinpartition f p) (Function.update p m (HAdd …
        -/
        ext v
        /-
          case refl.H
          𝕜 : Type u_1
          inst✝⁶ : NontriviallyNormedField 𝕜
          E : Type u_2
          inst✝⁵ : NormedAddCommGroup E
          inst✝⁴ : NormedSpace 𝕜 E
          F : Type u_3
          inst✝³ : NormedAddCommGroup F
          inst✝² : NormedSpace 𝕜 F
          G : Type u_4
          inst✝¹ : NormedAddCommGroup G
          inst✝ : NormedSpace 𝕜 G
          s : Set E
          t : Set F
          q✝ : F → FormalMultilinearSeries 𝕜 F G
          p✝ : E → FormalMultilinearSeries 𝕜 E F
          n : Nat
          c : OrderedFinpartition n
          f : ContinuousMultilinearMap 𝕜 (fun i => F) G
          p : (i : Fin c.length) → ContinuousMultilinearMap 𝕜 (fun i => E) F
          m : Fin c.length
          q q' : ContinuousMultilinearMap 𝕜 (fun i => E) F
          v : Fin n → E
          ⊢ Eq (((fun p => c.compAlongOrderedFinpartition f p) (Function.update p m (HAd …
        -/
        simp [applyOrderedFinpartition_update_left]
        /-
          🎉 no goals
        -/
      map_update_smul' := by
        /-
          𝕜 : Type u_1
          inst✝⁶ : NontriviallyNormedField 𝕜
          E : Type u_2
          inst✝⁵ : NormedAddCommGroup E
          inst✝⁴ : NormedSpace 𝕜 E
          F : Type u_3
          inst✝³ : NormedAddCommGroup F
          inst✝² : NormedSpace 𝕜 F
          G : Type u_4
          inst✝¹ : NormedAddCommGroup G
          inst✝ : NormedSpace 𝕜 G
          s : Set E
          t : Set F
          q : F → FormalMultilinearSeries 𝕜 F G
          p : E → FormalMultilinearSeries 𝕜 E F
          n : Nat
          c : OrderedFinpartition n
          f : ContinuousMultilinearMap 𝕜 (fun i => F) G
          ⊢ ∀ [inst : DecidableEq (Fin c.length)] (m : (i : Fin c.length) → ContinuousMu …
        -/
        intro inst p m a q
        /-
          𝕜 : Type u_1
          inst✝⁶ : NontriviallyNormedField 𝕜
          E : Type u_2
          inst✝⁵ : NormedAddCommGroup E
          inst✝⁴ : NormedSpace 𝕜 E
          F : Type u_3
          inst✝³ : NormedAddCommGroup F
          inst✝² : NormedSpace 𝕜 F
          G : Type u_4
          inst✝¹ : NormedAddCommGroup G
          inst✝ : NormedSpace 𝕜 G
          s : Set E
          t : Set F
          q✝ : F → FormalMultilinearSeries 𝕜 F G
          p✝ : E → FormalMultilinearSeries 𝕜 E F
          n : Nat
          c : OrderedFinpartition n
          f : ContinuousMultilinearMap 𝕜 (fun i => F) G
          inst : DecidableEq (Fin c.length)
          p : (i : Fin c.length) → ContinuousMultilinearMap 𝕜 (fun i => E) F
          m : Fin c.length
          a : 𝕜
          q : ContinuousMultilinearMap 𝕜 (fun i => E) F
          ⊢ Eq ((fun p => c.compAlongOrderedFinpartition f p) (Function.update p m (HSMu …
        -/
        cases Subsingleton.elim ‹_› (instDecidableEqFin _)
        /-
          case refl
          𝕜 : Type u_1
          inst✝⁶ : NontriviallyNormedField 𝕜
          E : Type u_2
          inst✝⁵ : NormedAddCommGroup E
          inst✝⁴ : NormedSpace 𝕜 E
          F : Type u_3
          inst✝³ : NormedAddCommGroup F
          inst✝² : NormedSpace 𝕜 F
          G : Type u_4
          inst✝¹ : NormedAddCommGroup G
          inst✝ : NormedSpace 𝕜 G
          s : Set E
          t : Set F
          q✝ : F → FormalMultilinearSeries 𝕜 F G
          p✝ : E → FormalMultilinearSeries 𝕜 E F
          n : Nat
          c : OrderedFinpartition n
          f : ContinuousMultilinearMap 𝕜 (fun i => F) G
          p : (i : Fin c.length) → ContinuousMultilinearMap 𝕜 (fun i => E) F
          m : Fin c.length
          a : 𝕜
          q : ContinuousMultilinearMap 𝕜 (fun i => E) F
          ⊢ Eq ((fun p => c.compAlongOrderedFinpartition f p) (Function.update p m (HSMu …
        -/
        ext v
        /-
          case refl.H
          𝕜 : Type u_1
          inst✝⁶ : NontriviallyNormedField 𝕜
          E : Type u_2
          inst✝⁵ : NormedAddCommGroup E
          inst✝⁴ : NormedSpace 𝕜 E
          F : Type u_3
          inst✝³ : NormedAddCommGroup F
          inst✝² : NormedSpace 𝕜 F
          G : Type u_4
          inst✝¹ : NormedAddCommGroup G
          inst✝ : NormedSpace 𝕜 G
          s : Set E
          t : Set F
          q✝ : F → FormalMultilinearSeries 𝕜 F G
          p✝ : E → FormalMultilinearSeries 𝕜 E F
          n : Nat
          c : OrderedFinpartition n
          f : ContinuousMultilinearMap 𝕜 (fun i => F) G
          p : (i : Fin c.length) → ContinuousMultilinearMap 𝕜 (fun i => E) F
          m : Fin c.length
          a : 𝕜
          q : ContinuousMultilinearMap 𝕜 (fun i => E) F
          v : Fin n → E
          ⊢ Eq (((fun p => c.compAlongOrderedFinpartition f p) (Function.update p m (HSM …
        -/
        simp [applyOrderedFinpartition_update_left] }
        /-
          🎉 no goals
        -/
  map_add' _ _ := rfl
  map_smul' _ _ :=  rfl


variable (𝕜 E F G) in
/-- Bundled version of `compAlongOrderedFinpartition`, depending continuously linearly on `f`
and continuously multilinearly on `p`.-/
noncomputable def compAlongOrderedFinpartitionL :
    (F [×c.length]→L[𝕜] G) →L[𝕜]
      ContinuousMultilinearMap 𝕜 (fun i ↦ E[×c.partSize i]→L[𝕜] F) (E[×n]→L[𝕜] G) := by
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    G : Type u_4
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    s : Set E
    t : Set F
    q : F → FormalMultilinearSeries 𝕜 F G
    p : E → FormalMultilinearSeries 𝕜 E F
    n : Nat
    c : OrderedFinpartition n
    ⊢ ContinuousLinearMap (RingHom.id 𝕜) (ContinuousMultilinearMap 𝕜 (fun i => F)  …
  -/
  refine MultilinearMap.mkContinuousLinear c.compAlongOrderedFinpartitionₗ 1 fun f p ↦ ?_
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    G : Type u_4
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    s : Set E
    t : Set F
    q : F → FormalMultilinearSeries 𝕜 F G
    p✝ : E → FormalMultilinearSeries 𝕜 E F
    n : Nat
    c : OrderedFinpartition n
    f : ContinuousMultilinearMap 𝕜 (fun i => F) G
    p : (i : Fin c.length) → ContinuousMultilinearMap 𝕜 (fun i => E) F
    ⊢ LE.le (Norm.norm ((c.compAlongOrderedFinpartitionₗ f) p)) (HMul.hMul (HMul.h …
  -/
  simp only [one_mul, compAlongOrderedFinpartitionₗ_apply_apply]
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    G : Type u_4
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    s : Set E
    t : Set F
    q : F → FormalMultilinearSeries 𝕜 F G
    p✝ : E → FormalMultilinearSeries 𝕜 E F
    n : Nat
    c : OrderedFinpartition n
    f : ContinuousMultilinearMap 𝕜 (fun i => F) G
    p : (i : Fin c.length) → ContinuousMultilinearMap 𝕜 (fun i => E) F
    ⊢ LE.le (Norm.norm (c.compAlongOrderedFinpartition f p)) (HMul.hMul (Norm.norm …
  -/
  apply norm_compAlongOrderedFinpartition_le
  /-
    🎉 no goals
  -/


@[simp] lemma compAlongOrderedFinpartitionL_apply (f : F [×c.length]→L[𝕜] G)
    (p : ∀ (i : Fin c.length), E[×c.partSize i]→L[𝕜] F) :
    c.compAlongOrderedFinpartitionL 𝕜 E F G f p = c.compAlongOrderedFinpartition f p := rfl


theorem norm_compAlongOrderedFinpartitionL_le :
    set_option maxSynthPendingDepth 2 in
    ‖c.compAlongOrderedFinpartitionL 𝕜 E F G‖ ≤ 1 :=
  MultilinearMap.mkContinuousLinear_norm_le _ zero_le_one _


/-- Given two formal multilinear series `q` and `p` and a composition `c` of `n`, one may
form a continuous multilinear map in `n` variables by applying the right coefficient of `p` to each
block of the composition, and then applying `q c.length` to the resulting vector. It is
called `q.compAlongComposition p c`. -/
def compAlongOrderedFinpartition {n : ℕ} (q : FormalMultilinearSeries 𝕜 F G)
    (p : FormalMultilinearSeries 𝕜 E F) (c : OrderedFinpartition n) :
    E [×n]→L[𝕜] G :=
  c.compAlongOrderedFinpartition (q c.length) (fun m ↦ p (c.partSize m))


@[simp]
theorem compAlongOrderedFinpartition_apply {n : ℕ} (q : FormalMultilinearSeries 𝕜 F G)
    (p : FormalMultilinearSeries 𝕜 E F) (c : OrderedFinpartition n) (v : Fin n → E) :
    (q.compAlongOrderedFinpartition p c) v =
      q c.length (c.applyOrderedFinpartition (fun m ↦ (p (c.partSize m))) v) :=
  rfl


/-- Taylor formal composition of two formal multilinear series. The `n`-th coefficient in the
composition is defined to be the sum of `q.compAlongOrderedFinpartition p c` over all
ordered partitions of `n`.
In other words, this term (as a multilinear function applied to `v₀, ..., vₙ₋₁`) is
`∑'_{k} ∑'_{I₀ ⊔ ... ⊔ Iₖ₋₁ = {0, ..., n-1}} qₖ (p_{i₀} (...), ..., p_{iₖ₋₁} (...))`, where
`iₘ` is the size of `Iₘ` and one puts all variables of `Iₘ` as arguments to `p_{iₘ}`, in
increasing order. The sets `I₀, ..., Iₖ₋₁` are ordered so that `max I₀ < max I₁ < ... < max Iₖ₋₁`.

This definition is chosen so that the `n`-th derivative of `g ∘ f` is the Taylor composition of
the iterated derivatives of `g` and of `f`.

Not to be confused with another notion of composition for formal multilinear series, called just
`FormalMultilinearSeries.comp`, appearing in the composition of analytic functions.
-/
protected noncomputable def taylorComp
    (q : FormalMultilinearSeries 𝕜 F G) (p : FormalMultilinearSeries 𝕜 E F) :
    FormalMultilinearSeries 𝕜 E G :=
  fun n ↦ ∑ c : OrderedFinpartition n, q.compAlongOrderedFinpartition p c


theorem analyticOn_taylorComp
    (hq : ∀ (n : ℕ), AnalyticOn 𝕜 (fun x ↦ q x n) t)
    (hp : ∀ n, AnalyticOn 𝕜 (fun x ↦ p x n) s) {f : E → F}
    (hf : AnalyticOn 𝕜 f s) (h : MapsTo f s t) (n : ℕ) :
    AnalyticOn 𝕜 (fun x ↦ (q (f x)).taylorComp (p x) n) s := by
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    G : Type u_4
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    s : Set E
    t : Set F
    q : F → FormalMultilinearSeries 𝕜 F G
    p : E → FormalMultilinearSeries 𝕜 E F
    hq : ∀ (n : Nat), AnalyticOn 𝕜 (fun x => q x n) t
    hp : ∀ (n : Nat), AnalyticOn 𝕜 (fun x => p x n) s
    f : E → F
    hf : AnalyticOn 𝕜 f s
    h : Set.MapsTo f s t
    n : Nat
    ⊢ AnalyticOn 𝕜 (fun x => (q (f x)).taylorComp (p x) n) s
  -/
  apply Finset.analyticOn_sum _ (fun c _ ↦ ?_)
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    G : Type u_4
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    s : Set E
    t : Set F
    q : F → FormalMultilinearSeries 𝕜 F G
    p : E → FormalMultilinearSeries 𝕜 E F
    hq : ∀ (n : Nat), AnalyticOn 𝕜 (fun x => q x n) t
    hp : ∀ (n : Nat), AnalyticOn 𝕜 (fun x => p x n) s
    f : E → F
    hf : AnalyticOn 𝕜 f s
    h : Set.MapsTo f s t
    n : Nat
    c : OrderedFinpartition n
    x✝ : Membership.mem Finset.univ c
    ⊢ AnalyticOn 𝕜 (fun z => (q (f z)).compAlongOrderedFinpartition (p z) c) s
  -/
  let B := c.compAlongOrderedFinpartitionL 𝕜 E F G
  change AnalyticOn 𝕜
    ((fun p ↦ B p.1 p.2) ∘ (fun x ↦ (q (f x) c.length, fun m ↦ p x (c.partSize m)))) s
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    G : Type u_4
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    s : Set E
    t : Set F
    q : F → FormalMultilinearSeries 𝕜 F G
    p : E → FormalMultilinearSeries 𝕜 E F
    hq : ∀ (n : Nat), AnalyticOn 𝕜 (fun x => q x n) t
    hp : ∀ (n : Nat), AnalyticOn 𝕜 (fun x => p x n) s
    f : E → F
    hf : AnalyticOn 𝕜 f s
    h : Set.MapsTo f s t
    n : Nat
    c : OrderedFinpartition n
    x✝ : Membership.mem Finset.univ c
    B : ContinuousLinearMap (RingHom.id 𝕜) (ContinuousMultilinearMap 𝕜 (fun i => F …
    ⊢ AnalyticOn 𝕜 (Function.comp (fun p => (B p.1) p.2) fun x => { fst := q (f x) …
  -/
  apply B.analyticOnNhd_uncurry_of_multilinear.comp_analyticOn ?_ (mapsTo_univ _ _)
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    G : Type u_4
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    s : Set E
    t : Set F
    q : F → FormalMultilinearSeries 𝕜 F G
    p : E → FormalMultilinearSeries 𝕜 E F
    hq : ∀ (n : Nat), AnalyticOn 𝕜 (fun x => q x n) t
    hp : ∀ (n : Nat), AnalyticOn 𝕜 (fun x => p x n) s
    f : E → F
    hf : AnalyticOn 𝕜 f s
    h : Set.MapsTo f s t
    n : Nat
    c : OrderedFinpartition n
    x✝ : Membership.mem Finset.univ c
    B : ContinuousLinearMap (RingHom.id 𝕜) (ContinuousMultilinearMap 𝕜 (fun i => F …
    ⊢ AnalyticOn 𝕜 (fun x => { fst := q (f x) c.length, snd := fun m => p x (c.par …
  -/
  apply AnalyticOn.prod
    /-
      case hf
      𝕜 : Type u_1
      inst✝⁶ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace 𝕜 E
      F : Type u_3
      inst✝³ : NormedAddCommGroup F
      inst✝² : NormedSpace 𝕜 F
      G : Type u_4
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace 𝕜 G
      s : Set E
      t : Set F
      q : F → FormalMultilinearSeries 𝕜 F G
      p : E → FormalMultilinearSeries 𝕜 E F
      hq : ∀ (n : Nat), AnalyticOn 𝕜 (fun x => q x n) t
      hp : ∀ (n : Nat), AnalyticOn 𝕜 (fun x => p x n) s
      f : E → F
      hf : AnalyticOn 𝕜 f s
      h : Set.MapsTo f s t
      n : Nat
      c : OrderedFinpartition n
      x✝ : Membership.mem Finset.univ c
      B : ContinuousLinearMap (RingHom.id 𝕜) (ContinuousMultilinearMap 𝕜 (fun i => F …
      ⊢ AnalyticOn 𝕜 (fun x => q (f x) c.length) s
    -/
  · exact (hq c.length).comp hf h
    /-
      🎉 no goals
    -/
    /-
      case hg
      𝕜 : Type u_1
      inst✝⁶ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace 𝕜 E
      F : Type u_3
      inst✝³ : NormedAddCommGroup F
      inst✝² : NormedSpace 𝕜 F
      G : Type u_4
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace 𝕜 G
      s : Set E
      t : Set F
      q : F → FormalMultilinearSeries 𝕜 F G
      p : E → FormalMultilinearSeries 𝕜 E F
      hq : ∀ (n : Nat), AnalyticOn 𝕜 (fun x => q x n) t
      hp : ∀ (n : Nat), AnalyticOn 𝕜 (fun x => p x n) s
      f : E → F
      hf : AnalyticOn 𝕜 f s
      h : Set.MapsTo f s t
      n : Nat
      c : OrderedFinpartition n
      x✝ : Membership.mem Finset.univ c
      B : ContinuousLinearMap (RingHom.id 𝕜) (ContinuousMultilinearMap 𝕜 (fun i => F …
      ⊢ AnalyticOn 𝕜 (fun x m => p x (c.partSize m)) s
    -/
  · exact AnalyticOn.pi (fun i ↦ hp _)
    /-
      🎉 no goals
    -/


/-- Composing two formal multilinear series `q` and `p` along an ordered partition extended by a
new atom to the left corresponds to applying `p 1` on the first coordinates, and the initial
ordered partition on the other coordinates.
This is one of the terms that appears when differentiating in the Faa di Bruno
formula, going from step `m` to step `m + 1`. -/
private lemma faaDiBruno_aux1 {m : ℕ} (q : FormalMultilinearSeries 𝕜 F G)
    (p : FormalMultilinearSeries 𝕜 E F) (c : OrderedFinpartition m) :
    (q.compAlongOrderedFinpartition p (c.extend none)).curryLeft =
    ((c.compAlongOrderedFinpartitionL 𝕜 E F G).flipMultilinear fun i ↦ p (c.partSize i)).comp
      ((q (c.length + 1)).curryLeft.comp ((continuousMultilinearCurryFin1 𝕜 E F) (p 1))) := by
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    G : Type u_4
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    m : Nat
    q : FormalMultilinearSeries 𝕜 F G
    p : FormalMultilinearSeries 𝕜 E F
    c : OrderedFinpartition m
    ⊢ Eq (q.compAlongOrderedFinpartition p (c.extend Option.none)).curryLeft (((Or …
  -/
  ext e v
  simp only [Nat.succ_eq_add_one, OrderedFinpartition.extend, extendLeft,
    ContinuousMultilinearMap.curryLeft_apply,
    FormalMultilinearSeries.compAlongOrderedFinpartition_apply, applyOrderedFinpartition_apply,
    ContinuousLinearMap.coe_comp', comp_apply, continuousMultilinearCurryFin1_apply,
    Matrix.zero_empty, ContinuousLinearMap.flipMultilinear_apply_apply,
    compAlongOrderedFinpartitionL_apply, compAlongOrderFinpartition_apply]
  /-
    case h.H
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    G : Type u_4
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    m : Nat
    q : FormalMultilinearSeries 𝕜 F G
    p : FormalMultilinearSeries 𝕜 E F
    c : OrderedFinpartition m
    e : E
    v : Fin m → E
    ⊢ Eq ((q (HAdd.hAdd c.length 1)) fun m_1 => (p (Fin.cons 1 c.partSize m_1)) (F …
  -/
  congr
  /-
    case h.H.h.e_6.h
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    G : Type u_4
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    m : Nat
    q : FormalMultilinearSeries 𝕜 F G
    p : FormalMultilinearSeries 𝕜 E F
    c : OrderedFinpartition m
    e : E
    v : Fin m → E
    ⊢ Eq (fun m_1 => (p (Fin.cons 1 c.partSize m_1)) (Function.comp (Fin.cons e v) …
  -/
  ext j
  /-
    case h.H.h.e_6.h.h
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    G : Type u_4
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    m : Nat
    q : FormalMultilinearSeries 𝕜 F G
    p : FormalMultilinearSeries 𝕜 E F
    c : OrderedFinpartition m
    e : E
    v : Fin m → E
    j : Fin (HAdd.hAdd c.length 1)
    ⊢ Eq ((p (Fin.cons 1 c.partSize j)) (Function.comp (Fin.cons e v) (Fin.cases ( …
  -/
  exact Fin.cases rfl (fun i ↦ rfl) j
  /-
    🎉 no goals
  -/


/-- Composing a formal multilinear series with an ordered partition extended by adding a left point
to an already existing atom of index `i` corresponds to updating the `i`th block,
using `p (c.partSize i + 1)` instead of `p (c.partSize i)` there.
This is one of the terms that appears when differentiating in the Faa di Bruno
formula, going from step `m` to step `m + 1`. -/
private lemma faaDiBruno_aux2 {m : ℕ} (q : FormalMultilinearSeries 𝕜 F G)
    (p : FormalMultilinearSeries 𝕜 E F) (c : OrderedFinpartition m) (i : Fin c.length) :
    (q.compAlongOrderedFinpartition p (c.extend (some i))).curryLeft =
    ((c.compAlongOrderedFinpartitionL 𝕜 E F G (q c.length)).toContinuousLinearMap
      (fun i ↦ p (c.partSize i)) i).comp (p (c.partSize i + 1)).curryLeft := by
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    G : Type u_4
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    m : Nat
    q : FormalMultilinearSeries 𝕜 F G
    p : FormalMultilinearSeries 𝕜 E F
    c : OrderedFinpartition m
    i : Fin c.length
    ⊢ Eq (q.compAlongOrderedFinpartition p (c.extend (Option.some i))).curryLeft ( …
  -/
  ext e v
  simp? [OrderedFinpartition.extend, extendMiddle, applyOrderedFinpartition_apply] says
    simp only [Nat.succ_eq_add_one, OrderedFinpartition.extend, extendMiddle,
      ContinuousMultilinearMap.curryLeft_apply,
      FormalMultilinearSeries.compAlongOrderedFinpartition_apply, applyOrderedFinpartition_apply,
      ContinuousLinearMap.coe_comp', comp_apply,
      ContinuousMultilinearMap.toContinuousLinearMap_apply, compAlongOrderedFinpartitionL_apply,
      compAlongOrderFinpartition_apply]
  /-
    case h.H
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    G : Type u_4
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    m : Nat
    q : FormalMultilinearSeries 𝕜 F G
    p : FormalMultilinearSeries 𝕜 E F
    c : OrderedFinpartition m
    i : Fin c.length
    e : E
    v : Fin m → E
    ⊢ Eq ((q c.length) fun m_1 => (p (Function.update c.partSize i (HAdd.hAdd (c.p …
  -/
  congr
  /-
    case h.H.h.e_6.h
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    G : Type u_4
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    m : Nat
    q : FormalMultilinearSeries 𝕜 F G
    p : FormalMultilinearSeries 𝕜 E F
    c : OrderedFinpartition m
    i : Fin c.length
    e : E
    v : Fin m → E
    ⊢ Eq (fun m_1 => (p (Function.update c.partSize i (HAdd.hAdd (c.partSize i) 1) …
  -/
  ext j
  /-
    case h.H.h.e_6.h.h
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    G : Type u_4
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    m : Nat
    q : FormalMultilinearSeries 𝕜 F G
    p : FormalMultilinearSeries 𝕜 E F
    c : OrderedFinpartition m
    i : Fin c.length
    e : E
    v : Fin m → E
    j : Fin c.length
    ⊢ Eq ((p (Function.update c.partSize i (HAdd.hAdd (c.partSize i) 1) j)) (Funct …
  -/
  rcases eq_or_ne j i with rfl | hij
  · simp only [↓reduceDIte, update_self, ContinuousMultilinearMap.curryLeft_apply,
      Nat.succ_eq_add_one]
    /-
      case h.H.h.e_6.h.h.inl
      𝕜 : Type u_1
      inst✝⁶ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace 𝕜 E
      F : Type u_3
      inst✝³ : NormedAddCommGroup F
      inst✝² : NormedSpace 𝕜 F
      G : Type u_4
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace 𝕜 G
      m : Nat
      q : FormalMultilinearSeries 𝕜 F G
      p : FormalMultilinearSeries 𝕜 E F
      c : OrderedFinpartition m
      e : E
      v : Fin m → E
      j : Fin c.length
      ⊢ Eq ((p (Function.update c.partSize j (HAdd.hAdd (c.partSize j) 1) j)) (Funct …
    -/
    apply FormalMultilinearSeries.congr _ (by simp)
    /-
      case h.H.h.e_6.h.h.inl
      𝕜 : Type u_1
      inst✝⁶ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace 𝕜 E
      F : Type u_3
      inst✝³ : NormedAddCommGroup F
      inst✝² : NormedSpace 𝕜 F
      G : Type u_4
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace 𝕜 G
      m : Nat
      q : FormalMultilinearSeries 𝕜 F G
      p : FormalMultilinearSeries 𝕜 E F
      c : OrderedFinpartition m
      e : E
      v : Fin m → E
      j : Fin c.length
      ⊢ ∀ (i : Nat) (him : LT.lt i (Function.update c.partSize j (HAdd.hAdd (c.partS …
    -/
    intro a ha h'a
    match a with
    | 0 => simp
    | a + 1 => simp [cons]
    /-
      case h.H.h.e_6.h.h.inr
      𝕜 : Type u_1
      inst✝⁶ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace 𝕜 E
      F : Type u_3
      inst✝³ : NormedAddCommGroup F
      inst✝² : NormedSpace 𝕜 F
      G : Type u_4
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace 𝕜 G
      m : Nat
      q : FormalMultilinearSeries 𝕜 F G
      p : FormalMultilinearSeries 𝕜 E F
      c : OrderedFinpartition m
      i : Fin c.length
      e : E
      v : Fin m → E
      j : Fin c.length
      hij : Ne j i
      ⊢ Eq ((p (Function.update c.partSize i (HAdd.hAdd (c.partSize i) 1) j)) (Funct …
    -/
  · simp only [hij, ↓reduceDIte, ne_eq, not_false_eq_true, update_of_ne]
    /-
      case h.H.h.e_6.h.h.inr
      𝕜 : Type u_1
      inst✝⁶ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace 𝕜 E
      F : Type u_3
      inst✝³ : NormedAddCommGroup F
      inst✝² : NormedSpace 𝕜 F
      G : Type u_4
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace 𝕜 G
      m : Nat
      q : FormalMultilinearSeries 𝕜 F G
      p : FormalMultilinearSeries 𝕜 E F
      c : OrderedFinpartition m
      i : Fin c.length
      e : E
      v : Fin m → E
      j : Fin c.length
      hij : Ne j i
      ⊢ Eq ((p (Function.update c.partSize i (HAdd.hAdd (c.partSize i) 1) j)) (Funct …
    -/
    apply FormalMultilinearSeries.congr _ (by simp [hij])
    /-
      case h.H.h.e_6.h.h.inr
      𝕜 : Type u_1
      inst✝⁶ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace 𝕜 E
      F : Type u_3
      inst✝³ : NormedAddCommGroup F
      inst✝² : NormedSpace 𝕜 F
      G : Type u_4
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace 𝕜 G
      m : Nat
      q : FormalMultilinearSeries 𝕜 F G
      p : FormalMultilinearSeries 𝕜 E F
      c : OrderedFinpartition m
      i : Fin c.length
      e : E
      v : Fin m → E
      j : Fin c.length
      hij : Ne j i
      ⊢ ∀ (i_1 : Nat) (him : LT.lt i_1 (Function.update c.partSize i (HAdd.hAdd (c.p …
    -/
    simp
    /-
      🎉 no goals
    -/


/-- *Faa di Bruno* formula: If two functions `g` and `f` have Taylor series up to `n` given by
`q` and `p`, then `g ∘ f` also has a Taylor series, given by `q.taylorComp p`. -/
theorem HasFTaylorSeriesUpToOn.comp {n : WithTop ℕ∞} {g : F → G} {f : E → F}
    (hg : HasFTaylorSeriesUpToOn n g q t) (hf : HasFTaylorSeriesUpToOn n f p s) (h : MapsTo f s t) :
    HasFTaylorSeriesUpToOn n (g ∘ f) (fun x ↦ (q (f x)).taylorComp (p x)) s := by
  /- One has to check that the `m+1`-th term is the derivative of the `m`-th term. The `m`-th term
  is a sum, that one can differentiate term by term. Each term is a linear map into continuous
  multilinear maps, applied to parts of `p` and `q`. One knows how to differentiate such a map,
  thanks to `HasFDerivWithinAt.linear_multilinear_comp`. The terms that show up are matched, using
  `faaDiBruno_aux1` and `faaDiBruno_aux2`, with terms of the same form at order `m+1`. Then, one
  needs to check that one gets each term once and exactly once, which is given by the bijection
  `OrderedFinpartition.extendEquiv m`. -/
  classical
  constructor
  · intro x hx
    simp [FormalMultilinearSeries.taylorComp, default, HasFTaylorSeriesUpToOn.zero_eq' hg (h hx)]
  · intro m hm x hx
    have A (c : OrderedFinpartition m) :
      HasFDerivWithinAt (fun x ↦ (q (f x)).compAlongOrderedFinpartition (p x) c)
        (∑ i : Option (Fin c.length),
          ((q (f x)).compAlongOrderedFinpartition (p x) (c.extend i)).curryLeft) s x := by
      let B := c.compAlongOrderedFinpartitionL 𝕜 E F G
      change HasFDerivWithinAt (fun y ↦ B (q (f y) c.length) (fun i ↦ p y (c.partSize i)))
        (∑ i : Option (Fin c.length),
          ((q (f x)).compAlongOrderedFinpartition (p x) (c.extend i)).curryLeft) s x
      have cm : (c.length : WithTop ℕ∞) ≤ m := mod_cast OrderedFinpartition.length_le c
      have cp i : (c.partSize i : WithTop ℕ∞) ≤ m := by
        exact_mod_cast OrderedFinpartition.partSize_le c i
      have I i : HasFDerivWithinAt (fun x ↦ p x (c.partSize i))
          (p x (c.partSize i).succ).curryLeft s x :=
        hf.fderivWithin (c.partSize i) ((cp i).trans_lt hm) x hx
      have J : HasFDerivWithinAt (fun x ↦ q x c.length) (q (f x) c.length.succ).curryLeft
        t (f x) := hg.fderivWithin c.length (cm.trans_lt hm) (f x) (h hx)
      have K : HasFDerivWithinAt f ((continuousMultilinearCurryFin1 𝕜 E F) (p x 1)) s x :=
        hf.hasFDerivWithinAt (le_trans (mod_cast Nat.le_add_left 1 m)
          (ENat.add_one_natCast_le_withTop_of_lt hm)) hx
      convert HasFDerivWithinAt.linear_multilinear_comp (J.comp x K h) I B
      simp only [B, Nat.succ_eq_add_one, Fintype.sum_option, comp_apply, faaDiBruno_aux1,
        faaDiBruno_aux2]
    have B : HasFDerivWithinAt (fun x ↦ (q (f x)).taylorComp (p x) m)
        (∑ c : OrderedFinpartition m, ∑ i : Option (Fin c.length),
          ((q (f x)).compAlongOrderedFinpartition (p x) (c.extend i)).curryLeft) s x :=
      HasFDerivWithinAt.sum (fun c _ ↦ A c)
    suffices ∑ c : OrderedFinpartition m, ∑ i : Option (Fin c.length),
          ((q (f x)).compAlongOrderedFinpartition (p x) (c.extend i)) =
        (q (f x)).taylorComp (p x) (m + 1) by
      rw [← this]
      convert B
      ext v
      simp only [Nat.succ_eq_add_one, Fintype.sum_option, ContinuousMultilinearMap.curryLeft_apply,
        ContinuousMultilinearMap.sum_apply, ContinuousMultilinearMap.add_apply,
        FormalMultilinearSeries.compAlongOrderedFinpartition_apply, ContinuousLinearMap.coe_sum',
        Finset.sum_apply, ContinuousLinearMap.add_apply]
    rw [Finset.sum_sigma']
    exact Fintype.sum_equiv (OrderedFinpartition.extendEquiv m) _ _ (fun p ↦ rfl)
  · intro m hm
    apply continuousOn_finset_sum _ (fun c _ ↦ ?_)
    let B := c.compAlongOrderedFinpartitionL 𝕜 E F G
    change ContinuousOn
      ((fun p ↦ B p.1 p.2) ∘ (fun x ↦ (q (f x) c.length, fun i ↦ p x (c.partSize i)))) s
    apply B.continuous_uncurry_of_multilinear.comp_continuousOn (ContinuousOn.prod ?_ ?_)
    · have : (c.length : WithTop ℕ∞) ≤ m := mod_cast OrderedFinpartition.length_le c
      exact (hg.cont c.length (this.trans hm)).comp hf.continuousOn h
    · apply continuousOn_pi.2 (fun i ↦ ?_)
      have : (c.partSize i : WithTop ℕ∞) ≤ m := by
        exact_mod_cast OrderedFinpartition.partSize_le c i
      exact hf.cont _ (this.trans hm)

