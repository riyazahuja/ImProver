/-- `smulAntidiagonal s t a` is the set of all pairs of an element in `s` and an
      element in `t` that scalar multiply to `a`.-/
@[to_additive "`vaddAntidiagonal s t a` is the set of all pairs of an element in `s` and an
      element in `t` that vector-add to `a`."]
def smulAntidiagonal (s : Set G) (t : Set P) (a : P) : Set (G × P) :=
  { x | x.1 ∈ s ∧ x.2 ∈ t ∧ x.1 • x.2 = a }


@[to_additive (attr := simp)]
theorem mem_smulAntidiagonal : x ∈ smulAntidiagonal s t a ↔ x.1 ∈ s ∧ x.2 ∈ t ∧ x.1 • x.2 = a :=
  Iff.rfl


@[to_additive]
theorem smulAntidiagonal_mono_left (h : s₁ ⊆ s₂) :
    smulAntidiagonal s₁ t a ⊆ smulAntidiagonal s₂ t a :=
  fun _ hx => ⟨h hx.1, hx.2.1, hx.2.2⟩


@[to_additive]
theorem smulAntidiagonal_mono_right (h : t₁ ⊆ t₂) :
    smulAntidiagonal s t₁ a ⊆ smulAntidiagonal s t₂ a := fun _ hx => ⟨hx.1, h hx.2.1, hx.2.2⟩


@[to_additive VAddAntidiagonal.fst_eq_fst_iff_snd_eq_snd]
theorem fst_eq_fst_iff_snd_eq_snd :
    (x : G × P).1 = (y : G × P).1 ↔ (x : G × P).2 = (y : G × P).2 :=
  ⟨fun h =>
    IsCancelSMul.left_cancel _ _ _
      (y.2.2.2.trans <| by
          /-
            G : Type u_1
            P : Type u_2
            s : Set G
            t : Set P
            a : P
            inst✝¹ : SMul G P
            inst✝ : IsCancelSMul G P
            x y : ↑(s.smulAntidiagonal t a)
            h : Eq (↑x).1 (↑y).1
            ⊢ Eq a (HSMul.hSMul (↑y).1 (↑x).2)
          -/
          rw [← h]
          /-
            G : Type u_1
            P : Type u_2
            s : Set G
            t : Set P
            a : P
            inst✝¹ : SMul G P
            inst✝ : IsCancelSMul G P
            x y : ↑(s.smulAntidiagonal t a)
            h : Eq (↑x).1 (↑y).1
            ⊢ Eq a (HSMul.hSMul (↑x).1 (↑x).2)
          -/
          exact x.2.2.2.symm).symm,
          /-
            🎉 no goals
          -/
    fun h =>
    IsCancelSMul.right_cancel _ _ _
      (y.2.2.2.trans <| by
          /-
            G : Type u_1
            P : Type u_2
            s : Set G
            t : Set P
            a : P
            inst✝¹ : SMul G P
            inst✝ : IsCancelSMul G P
            x y : ↑(s.smulAntidiagonal t a)
            h : Eq (↑x).2 (↑y).2
            ⊢ Eq a (HSMul.hSMul (↑x).1 (↑y).2)
          -/
          rw [← h]
          /-
            G : Type u_1
            P : Type u_2
            s : Set G
            t : Set P
            a : P
            inst✝¹ : SMul G P
            inst✝ : IsCancelSMul G P
            x y : ↑(s.smulAntidiagonal t a)
            h : Eq (↑x).2 (↑y).2
            ⊢ Eq a (HSMul.hSMul (↑x).1 (↑x).2)
          -/
          exact x.2.2.2.symm).symm⟩
          /-
            🎉 no goals
          -/


@[to_additive VAddAntidiagonal.eq_of_fst_eq_fst]
theorem eq_of_fst_eq_fst (h : (x : G × P).fst = (y : G × P).fst) : x = y :=
  Subtype.ext <| Prod.ext h <| fst_eq_fst_iff_snd_eq_snd.1 h


@[to_additive VAddAntidiagonal.eq_of_snd_eq_snd]
theorem eq_of_snd_eq_snd (h : (x : G × P).snd = (y : G × P).snd) : x = y :=
  Subtype.ext <| Prod.ext (fst_eq_fst_iff_snd_eq_snd.2 h) h


@[to_additive VAddAntidiagonal.eq_of_fst_le_fst_of_snd_le_snd]
theorem eq_of_fst_le_fst_of_snd_le_snd (h₁ : (x : G × P).1 ≤ (y : G × P).1)
    (h₂ : (x : G × P).2 ≤ (y : G × P).2) : x = y :=
  eq_of_fst_eq_fst <|
    h₁.eq_of_not_lt fun hlt =>
      (smul_lt_smul_of_lt_of_le hlt h₂).ne <|
        (mem_smulAntidiagonal.1 x.2).2.2.trans (mem_smulAntidiagonal.1 y.2).2.2.symm


@[to_additive VAddAntidiagonal.finite_of_isPWO]
theorem finite_of_isPWO (hs : s.IsPWO) (ht : t.IsPWO) (a) : (smulAntidiagonal s t a).Finite := by
  /-
    G : Type u_1
    P : Type u_2
    s : Set G
    t : Set P
    inst✝³ : PartialOrder G
    inst✝² : PartialOrder P
    inst✝¹ : SMul G P
    inst✝ : IsOrderedCancelSMul G P
    hs : s.IsPWO
    ht : t.IsPWO
    a : P
    ⊢ (s.smulAntidiagonal t a).Finite
  -/
  refine Set.not_infinite.1 fun h => ?_
  have h1 : (smulAntidiagonal s t a).PartiallyWellOrderedOn (Prod.fst ⁻¹'o (· ≤ ·)) := fun f hf =>
    hs (Prod.fst ∘ f) fun n => (mem_smulAntidiagonal.1 (hf n)).1
  have h2 : (smulAntidiagonal s t a).PartiallyWellOrderedOn (Prod.snd ⁻¹'o (· ≤ ·)) := fun f hf =>
    ht (Prod.snd ∘ f) fun n => (mem_smulAntidiagonal.1 (hf n)).2.1
  have isrfl : IsRefl (G × P) (Prod.fst ⁻¹'o fun x x_1 ↦ x ≤ x_1) := by
    refine { refl := ?refl }
    simp_all only [Order.Preimage, le_refl, Prod.forall, implies_true]
  have istrns : IsTrans (G × P) (Prod.fst ⁻¹'o fun x x_1 ↦ x ≤ x_1) := by
    refine { trans := ?trans }
    simp_all only [Order.Preimage, Prod.forall]
    exact fun a _ a_1 _ a_2 _ a_3 a_4 ↦ Preorder.le_trans a a_1 a_2 a_3 a_4
  obtain ⟨g, hg⟩ :=
    h1.exists_monotone_subseq (fun n => h.natEmbedding _ n) fun n => (h.natEmbedding _ n).2
  /-
    case intro
    G : Type u_1
    P : Type u_2
    s : Set G
    t : Set P
    inst✝³ : PartialOrder G
    inst✝² : PartialOrder P
    inst✝¹ : SMul G P
    inst✝ : IsOrderedCancelSMul G P
    hs : s.IsPWO
    ht : t.IsPWO
    a : P
    h : (s.smulAntidiagonal t a).Infinite
    h1 : (s.smulAntidiagonal t a).PartiallyWellOrderedOn (Order.Preimage Prod.fst  …
    h2 : (s.smulAntidiagonal t a).PartiallyWellOrderedOn (Order.Preimage Prod.snd  …
    isrfl : IsRefl (Prod G P) (Order.Preimage Prod.fst fun x x_1 => LE.le x x_1)
    istrns : IsTrans (Prod G P) (Order.Preimage Prod.fst fun x x_1 => LE.le x x_1)
    g : OrderEmbedding Nat Nat
    hg : ∀ (m n : Nat), LE.le m n → Order.Preimage Prod.fst (fun x1 x2 => LE.le x1 …
    ⊢ False
  -/
  obtain ⟨m, n, mn, h2'⟩ := h2 (fun x => (h.natEmbedding _) (g x)) fun n => (h.natEmbedding _ _).2
  /-
    case intro.intro.intro.intro
    G : Type u_1
    P : Type u_2
    s : Set G
    t : Set P
    inst✝³ : PartialOrder G
    inst✝² : PartialOrder P
    inst✝¹ : SMul G P
    inst✝ : IsOrderedCancelSMul G P
    hs : s.IsPWO
    ht : t.IsPWO
    a : P
    h : (s.smulAntidiagonal t a).Infinite
    h1 : (s.smulAntidiagonal t a).PartiallyWellOrderedOn (Order.Preimage Prod.fst  …
    h2 : (s.smulAntidiagonal t a).PartiallyWellOrderedOn (Order.Preimage Prod.snd  …
    isrfl : IsRefl (Prod G P) (Order.Preimage Prod.fst fun x x_1 => LE.le x x_1)
    istrns : IsTrans (Prod G P) (Order.Preimage Prod.fst fun x x_1 => LE.le x x_1)
    g : OrderEmbedding Nat Nat
    hg : ∀ (m n : Nat), LE.le m n → Order.Preimage Prod.fst (fun x1 x2 => LE.le x1 …
    m n : Nat
    mn : LT.lt m n
    h2' : Order.Preimage Prod.snd (fun x1 x2 => LE.le x1 x2) ↑((Set.Infinite.natEm …
    ⊢ False
  -/
  refine mn.ne (g.injective <| (h.natEmbedding _).injective ?_)
  /-
    case intro.intro.intro.intro
    G : Type u_1
    P : Type u_2
    s : Set G
    t : Set P
    inst✝³ : PartialOrder G
    inst✝² : PartialOrder P
    inst✝¹ : SMul G P
    inst✝ : IsOrderedCancelSMul G P
    hs : s.IsPWO
    ht : t.IsPWO
    a : P
    h : (s.smulAntidiagonal t a).Infinite
    h1 : (s.smulAntidiagonal t a).PartiallyWellOrderedOn (Order.Preimage Prod.fst  …
    h2 : (s.smulAntidiagonal t a).PartiallyWellOrderedOn (Order.Preimage Prod.snd  …
    isrfl : IsRefl (Prod G P) (Order.Preimage Prod.fst fun x x_1 => LE.le x x_1)
    istrns : IsTrans (Prod G P) (Order.Preimage Prod.fst fun x x_1 => LE.le x x_1)
    g : OrderEmbedding Nat Nat
    hg : ∀ (m n : Nat), LE.le m n → Order.Preimage Prod.fst (fun x1 x2 => LE.le x1 …
    m n : Nat
    mn : LT.lt m n
    h2' : Order.Preimage Prod.snd (fun x1 x2 => LE.le x1 x2) ↑((Set.Infinite.natEm …
    ⊢ Eq ((Set.Infinite.natEmbedding (s.smulAntidiagonal t a) h) (g m)) ((Set.Infi …
  -/
  exact eq_of_fst_le_fst_of_snd_le_snd (hg _ _ mn.le) h2'
  /-
    🎉 no goals
  -/


