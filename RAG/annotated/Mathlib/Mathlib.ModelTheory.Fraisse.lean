/-- The age of a structure `M` is the class of finitely-generated structures that embed into it. -/
def age (M : Type w) [L.Structure M] : Set (Bundled.{w} L.Structure) :=
  {N | Structure.FG L N ∧ Nonempty (N ↪[L] M)}


/-- A class `K` has the hereditary property when all finitely-generated structures that embed into
  structures in `K` are also in `K`. -/
def Hereditary : Prop :=
  ∀ M : Bundled.{w} L.Structure, M ∈ K → L.age M ⊆ K


/-- A class `K` has the joint embedding property when for every `M`, `N` in `K`, there is another
  structure in `K` into which both `M` and `N` embed. -/
def JointEmbedding : Prop :=
  DirectedOn (fun M N : Bundled.{w} L.Structure => Nonempty (M ↪[L] N)) K


/-- A class `K` has the amalgamation property when for any pair of embeddings of a structure `M` in
  `K` into other structures in `K`, those two structures can be embedded into a fourth structure in
  `K` such that the resulting square of embeddings commutes. -/
def Amalgamation : Prop :=
  ∀ (M N P : Bundled.{w} L.Structure) (MN : M ↪[L] N) (MP : M ↪[L] P),
    M ∈ K → N ∈ K → P ∈ K → ∃ (Q : Bundled.{w} L.Structure) (NQ : N ↪[L] Q) (PQ : P ↪[L] Q),
      Q ∈ K ∧ NQ.comp MN = PQ.comp MP


/-- A Fraïssé class is a nonempty, essentially countable class of structures satisfying the
hereditary, joint embedding, and amalgamation properties. -/
class IsFraisse : Prop where
  is_nonempty : K.Nonempty
  FG : ∀ M : Bundled.{w} L.Structure, M ∈ K → Structure.FG L M
  is_essentially_countable : (Quotient.mk' '' K).Countable
  hereditary : Hereditary K
  jointEmbedding : JointEmbedding K
  amalgamation : Amalgamation K


theorem age.is_equiv_invariant (N P : Bundled.{w} L.Structure) (h : Nonempty (N ≃[L] P)) :
    N ∈ L.age M ↔ P ∈ L.age M :=
  and_congr h.some.fg_iff
    ⟨Nonempty.map fun x => Embedding.comp x h.some.symm.toEmbedding,
      Nonempty.map fun x => Embedding.comp x h.some.toEmbedding⟩


theorem Embedding.age_subset_age (MN : M ↪[L] N) : L.age M ⊆ L.age N := fun _ =>
  And.imp_right (Nonempty.map MN.comp)


theorem Equiv.age_eq_age (MN : M ≃[L] N) : L.age M = L.age N :=
  le_antisymm MN.toEmbedding.age_subset_age MN.symm.toEmbedding.age_subset_age


theorem Structure.FG.mem_age_of_equiv {M N : Bundled L.Structure} (h : Structure.FG L M)
    (MN : Nonempty (M ≃[L] N)) : N ∈ L.age M :=
  ⟨MN.some.fg_iff.1 h, ⟨MN.some.symm.toEmbedding⟩⟩


theorem Hereditary.is_equiv_invariant_of_fg (h : Hereditary K)
    (fg : ∀ M : Bundled.{w} L.Structure, M ∈ K → Structure.FG L M) (M N : Bundled.{w} L.Structure)
    (hn : Nonempty (M ≃[L] N)) : M ∈ K ↔ N ∈ K :=
  ⟨fun MK => h M MK ((fg M MK).mem_age_of_equiv hn),
   fun NK => h N NK ((fg N NK).mem_age_of_equiv ⟨hn.some.symm⟩)⟩


theorem IsFraisse.is_equiv_invariant [h : IsFraisse K] {M N : Bundled.{w} L.Structure}
    (hn : Nonempty (M ≃[L] N)) : M ∈ K ↔ N ∈ K :=
  h.hereditary.is_equiv_invariant_of_fg h.FG M N hn


theorem age.nonempty : (L.age M).Nonempty :=
  ⟨Bundled.of (Substructure.closure L (∅ : Set M)),
    (fg_iff_structure_fg _).1 (fg_closure Set.finite_empty), ⟨Substructure.subtype _⟩⟩


theorem age.hereditary : Hereditary (L.age M) := fun _ hN _ hP => hN.2.some.age_subset_age hP


theorem age.jointEmbedding : JointEmbedding (L.age M) := fun _ hN _ hP =>
  ⟨Bundled.of (↥(hN.2.some.toHom.range ⊔ hP.2.some.toHom.range)),
    ⟨(fg_iff_structure_fg _).1 ((hN.1.range hN.2.some.toHom).sup (hP.1.range hP.2.some.toHom)),
      ⟨Substructure.subtype _⟩⟩,
    ⟨Embedding.comp (inclusion le_sup_left) hN.2.some.equivRange.toEmbedding⟩,
    ⟨Embedding.comp (inclusion le_sup_right) hP.2.some.equivRange.toEmbedding⟩⟩


                                                                 /-
                                                                   L : FirstOrder.Language
                                                                   K : Set (CategoryTheory.Bundled L.Structure)
                                                                   M : Type w
                                                                   inst✝¹ : L.Structure M
                                                                   N : Type w
                                                                   inst✝ : L.Structure N
                                                                   S : L.Substructure M
                                                                   fg : S.FG
                                                                   ⊢ L.Structure (Subtype fun x => Membership.mem S x)
                                                                 -/
theorem age.fg_substructure {S : L.Substructure M} (fg : S.FG) : Bundled.mk S ∈ L.age M := by
                                                                 /-
                                                                   🎉 no goals
                                                                 -/
  /-
    L : FirstOrder.Language
    M : Type w
    inst✝ : L.Structure M
    S : L.Substructure M
    fg : S.FG
    ⊢ Membership.mem (L.age M) { α := Subtype fun x => Membership.mem S x, str :=  …
  -/
  exact ⟨(Substructure.fg_iff_structure_fg _).1 fg, ⟨subtype _⟩⟩
  /-
    🎉 no goals
  -/


/-- Any class in the age of a structure has a representative which is a finitely generated
substructure. -/
theorem age.has_representative_as_substructure :
    ∀ C ∈ Quotient.mk' '' L.age M, ∃ V : {V : L.Substructure M // FG V},
       /-
         L : FirstOrder.Language
         K : Set (CategoryTheory.Bundled L.Structure)
         M : Type w
         inst✝¹ : L.Structure M
         N : Type w
         inst✝ : L.Structure N
         C : Quotient FirstOrder.Language.equivSetoid
         V : Subtype fun V => V.FG
         ⊢ L.Structure (Subtype fun x => Membership.mem (↑V) x)
       -/
      ⟦Bundled.mk V⟧ = C := by
       /-
         🎉 no goals
       -/
  /-
    L : FirstOrder.Language
    M : Type w
    inst✝ : L.Structure M
    ⊢ ∀ (C : Quotient FirstOrder.Language.equivSetoid), Membership.mem (Set.image  …
  -/
  rintro _ ⟨N, ⟨N_fg, ⟨N_incl⟩⟩, N_eq⟩
  /-
    case intro.intro.intro.intro
    L : FirstOrder.Language
    M : Type w
    inst✝ : L.Structure M
    C✝ : Quotient FirstOrder.Language.equivSetoid
    N : CategoryTheory.Bundled L.Structure
    N_eq : Eq (Quotient.mk' N) C✝
    N_fg : FirstOrder.Language.Structure.FG L ↑N
    N_incl : L.Embedding (↑N) M
    ⊢ Exists fun V => Eq (Quotient.mk FirstOrder.Language.equivSetoid { α := Subty …
  -/
  refine N_eq.symm ▸ ⟨⟨N_incl.toHom.range, ?_⟩, Quotient.sound ⟨N_incl.equivRange.symm⟩⟩
  /-
    case intro.intro.intro.intro
    L : FirstOrder.Language
    M : Type w
    inst✝ : L.Structure M
    C✝ : Quotient FirstOrder.Language.equivSetoid
    N : CategoryTheory.Bundled L.Structure
    N_eq : Eq (Quotient.mk' N) C✝
    N_fg : FirstOrder.Language.Structure.FG L ↑N
    N_incl : L.Embedding (↑N) M
    ⊢ N_incl.toHom.range.FG
  -/
  exact FG.range N_fg (Embedding.toHom N_incl)
  /-
    🎉 no goals
  -/


/-- The age of a countable structure is essentially countable (has countably many isomorphism
classes). -/
theorem age.countable_quotient [h : Countable M] : (Quotient.mk' '' L.age M).Countable := by
  classical
  refine (congr_arg _ (Set.ext <| Quotient.forall.2 fun N => ?_)).mp
    (countable_range fun s : Finset M => ⟦⟨closure L (s : Set M), inferInstance⟩⟧)
  constructor
  · rintro ⟨s, hs⟩
    use Bundled.of (closure L (s : Set M))
    exact ⟨⟨(fg_iff_structure_fg _).1 (fg_closure s.finite_toSet), ⟨Substructure.subtype _⟩⟩, hs⟩
  · simp only [mem_range, Quotient.eq]
    rintro ⟨P, ⟨⟨s, hs⟩, ⟨PM⟩⟩, hP2⟩
    have : P ≈ N := by apply Quotient.eq'.mp; rw [hP2]; rfl -- Porting note: added
    refine ⟨s.image PM, Setoid.trans (b := P) ?_ this⟩
    rw [← Embedding.coe_toHom, Finset.coe_image, closure_image PM.toHom, hs, ← Hom.range_eq_map]
    exact ⟨PM.equivRange.symm⟩


/-- The age of a direct limit of structures is the union of the ages of the structures. -/
-- @[simp] -- Porting note: cannot simplify itself
theorem age_directLimit {ι : Type w} [Preorder ι] [IsDirected ι (· ≤ ·)] [Nonempty ι]
    (G : ι → Type max w w') [∀ i, L.Structure (G i)] (f : ∀ i j, i ≤ j → G i ↪[L] G j)
    [DirectedSystem G fun i j h => f i j h] : L.age (DirectLimit G f) = ⋃ i : ι, L.age (G i) := by
  classical
  ext M
  simp only [mem_iUnion]
  constructor
  · rintro ⟨Mfg, ⟨e⟩⟩
    obtain ⟨s, hs⟩ := Mfg.range e.toHom
    let out := @Quotient.out _ (DirectLimit.setoid G f)
    obtain ⟨i, hi⟩ := Finset.exists_le (s.image (Sigma.fst ∘ out))
    have e' := (DirectLimit.of L ι G f i).equivRange.symm.toEmbedding
    refine ⟨i, Mfg, ⟨e'.comp ((Substructure.inclusion ?_).comp e.equivRange.toEmbedding)⟩⟩
    rw [← hs, closure_le]
    intro x hx
    refine ⟨f (out x).1 i (hi (out x).1 (Finset.mem_image_of_mem _ hx)) (out x).2, ?_⟩
    rw [Embedding.coe_toHom, DirectLimit.of_apply, @Quotient.mk_eq_iff_out _ (_),
      DirectLimit.equiv_iff G f _ (hi (out x).1 (Finset.mem_image_of_mem _ hx)),
      DirectedSystem.map_self]
  · rintro ⟨i, Mfg, ⟨e⟩⟩
    exact ⟨Mfg, ⟨Embedding.comp (DirectLimit.of L ι G f i) e⟩⟩


/-- Sufficient conditions for a class to be the age of a countably-generated structure. -/
theorem exists_cg_is_age_of (hn : K.Nonempty)
    (hc : (Quotient.mk' '' K).Countable)
    (fg : ∀ M : Bundled.{w} L.Structure, M ∈ K → Structure.FG L M) (hp : Hereditary K)
    (jep : JointEmbedding K) : ∃ M : Bundled.{w} L.Structure, Structure.CG L M ∧ L.age M = K := by
  /-
    L : FirstOrder.Language
    K : Set (CategoryTheory.Bundled L.Structure)
    hn : K.Nonempty
    hc : (Set.image Quotient.mk' K).Countable
    fg : ∀ (M : CategoryTheory.Bundled L.Structure), Membership.mem K M → FirstOrd …
    hp : FirstOrder.Language.Hereditary K
    jep : FirstOrder.Language.JointEmbedding K
    ⊢ Exists fun M => And (FirstOrder.Language.Structure.CG L ↑M) (Eq (L.age ↑M) K)
  -/
  obtain ⟨F, hF⟩ := hc.exists_eq_range (hn.image _)
  /-
    case intro
    L : FirstOrder.Language
    K : Set (CategoryTheory.Bundled L.Structure)
    hn : K.Nonempty
    hc : (Set.image Quotient.mk' K).Countable
    fg : ∀ (M : CategoryTheory.Bundled L.Structure), Membership.mem K M → FirstOrd …
    hp : FirstOrder.Language.Hereditary K
    jep : FirstOrder.Language.JointEmbedding K
    F : Nat → Quotient FirstOrder.Language.equivSetoid
    hF : Eq (Set.image Quotient.mk' K) (Set.range F)
    ⊢ Exists fun M => And (FirstOrder.Language.Structure.CG L ↑M) (Eq (L.age ↑M) K)
  -/
  simp only [Set.ext_iff, Quotient.forall, mem_image, mem_range, Quotient.eq'] at hF
  /-
    case intro
    L : FirstOrder.Language
    K : Set (CategoryTheory.Bundled L.Structure)
    hn : K.Nonempty
    hc : (Set.image Quotient.mk' K).Countable
    fg : ∀ (M : CategoryTheory.Bundled L.Structure), Membership.mem K M → FirstOrd …
    hp : FirstOrder.Language.Hereditary K
    jep : FirstOrder.Language.JointEmbedding K
    F : Nat → Quotient FirstOrder.Language.equivSetoid
    hF : ∀ (a : CategoryTheory.Bundled L.Structure), Iff (Exists fun x => And (Mem …
    ⊢ Exists fun M => And (FirstOrder.Language.Structure.CG L ↑M) (Eq (L.age ↑M) K)
  -/
  simp_rw [Quotient.eq_mk_iff_out] at hF
  have hF' : ∀ n : ℕ, (F n).out ∈ K := by
    intro n
    obtain ⟨P, hP1, hP2⟩ := (hF (F n).out).2 ⟨n, Setoid.refl _⟩
    -- Porting note: fix hP2 because `Quotient.out (Quotient.mk' x) ≈ a` was not simplified
    -- to `x ≈ a` in hF
    replace hP2 := Setoid.trans (Setoid.symm (Quotient.mk_out P)) hP2
    exact (hp.is_equiv_invariant_of_fg fg _ _ hP2).1 hP1
  /-
    case intro
    L : FirstOrder.Language
    K : Set (CategoryTheory.Bundled L.Structure)
    hn : K.Nonempty
    hc : (Set.image Quotient.mk' K).Countable
    fg : ∀ (M : CategoryTheory.Bundled L.Structure), Membership.mem K M → FirstOrd …
    hp : FirstOrder.Language.Hereditary K
    jep : FirstOrder.Language.JointEmbedding K
    F : Nat → Quotient FirstOrder.Language.equivSetoid
    hF : ∀ (a : CategoryTheory.Bundled L.Structure), Iff (Exists fun x => And (Mem …
    hF' : ∀ (n : Nat), Membership.mem K (F n).out
    ⊢ Exists fun M => And (FirstOrder.Language.Structure.CG L ↑M) (Eq (L.age ↑M) K)
  -/
  choose P hPK hP hFP using fun (N : K) (n : ℕ) => jep N N.2 (F (n + 1)).out (hF' _)
  /-
    case intro
    L : FirstOrder.Language
    K : Set (CategoryTheory.Bundled L.Structure)
    hn : K.Nonempty
    hc : (Set.image Quotient.mk' K).Countable
    fg : ∀ (M : CategoryTheory.Bundled L.Structure), Membership.mem K M → FirstOrd …
    hp : FirstOrder.Language.Hereditary K
    jep : FirstOrder.Language.JointEmbedding K
    F : Nat → Quotient FirstOrder.Language.equivSetoid
    hF : ∀ (a : CategoryTheory.Bundled L.Structure), Iff (Exists fun x => And (Mem …
    hF' : ∀ (n : Nat), Membership.mem K (F n).out
    P : ↑K → Nat → CategoryTheory.Bundled L.Structure
    hPK : ∀ (N : ↑K) (n : Nat), Membership.mem K (P N n)
    hP : ∀ (N : ↑K) (n : Nat), (fun M N => Nonempty (L.Embedding ↑M ↑N)) (↑N) (P N …
    hFP : ∀ (N : ↑K) (n : Nat), (fun M N => Nonempty (L.Embedding ↑M ↑N)) (F (HAdd …
    ⊢ Exists fun M => And (FirstOrder.Language.Structure.CG L ↑M) (Eq (L.age ↑M) K)
  -/
  let G : ℕ → K := @Nat.rec (fun _ => K) ⟨(F 0).out, hF' 0⟩ fun n N => ⟨P N n, hPK N n⟩
  -- Poting note: was
  -- let f : ∀ i j, i ≤ j → G i ↪[L] G j := DirectedSystem.natLeRec fun n => (hP _ n).some
  let f : ∀ (i j : ℕ), i ≤ j → (G i).val ↪[L] (G j).val := by
    refine DirectedSystem.natLERec (G' := fun i => (G i).val) (L := L) ?_
    dsimp only [G]
    exact fun n => (hP _ n).some
  have : DirectedSystem (fun n ↦ (G n).val) fun i j h ↦ ↑(f i j h) := by
    dsimp [f, G]; infer_instance
  /-
    case intro
    L : FirstOrder.Language
    K : Set (CategoryTheory.Bundled L.Structure)
    hn : K.Nonempty
    hc : (Set.image Quotient.mk' K).Countable
    fg : ∀ (M : CategoryTheory.Bundled L.Structure), Membership.mem K M → FirstOrd …
    hp : FirstOrder.Language.Hereditary K
    jep : FirstOrder.Language.JointEmbedding K
    F : Nat → Quotient FirstOrder.Language.equivSetoid
    hF : ∀ (a : CategoryTheory.Bundled L.Structure), Iff (Exists fun x => And (Mem …
    hF' : ∀ (n : Nat), Membership.mem K (F n).out
    P : ↑K → Nat → CategoryTheory.Bundled L.Structure
    hPK : ∀ (N : ↑K) (n : Nat), Membership.mem K (P N n)
    hP : ∀ (N : ↑K) (n : Nat), (fun M N => Nonempty (L.Embedding ↑M ↑N)) (↑N) (P N …
    hFP : ∀ (N : ↑K) (n : Nat), (fun M N => Nonempty (L.Embedding ↑M ↑N)) (F (HAdd …
    G : Nat → ↑K := Nat.rec ⟨(F 0).out, ⋯⟩ fun n N => ⟨P N n, ⋯⟩
    f : (i j : Nat) → LE.le i j → L.Embedding ↑↑(G i) ↑↑(G j) := FirstOrder.Langua …
    this : DirectedSystem (fun n => ↑↑(G n)) fun i j h => ⇑(f i j h)
    ⊢ Exists fun M => And (FirstOrder.Language.Structure.CG L ↑M) (Eq (L.age ↑M) K)
  -/
  refine ⟨Bundled.of (@DirectLimit L _ _ (fun n ↦ (G n).val) _ f _ _), ?_, ?_⟩
    /-
      case intro.refine_1
      L : FirstOrder.Language
      K : Set (CategoryTheory.Bundled L.Structure)
      hn : K.Nonempty
      hc : (Set.image Quotient.mk' K).Countable
      fg : ∀ (M : CategoryTheory.Bundled L.Structure), Membership.mem K M → FirstOrd …
      hp : FirstOrder.Language.Hereditary K
      jep : FirstOrder.Language.JointEmbedding K
      F : Nat → Quotient FirstOrder.Language.equivSetoid
      hF : ∀ (a : CategoryTheory.Bundled L.Structure), Iff (Exists fun x => And (Mem …
      hF' : ∀ (n : Nat), Membership.mem K (F n).out
      P : ↑K → Nat → CategoryTheory.Bundled L.Structure
      hPK : ∀ (N : ↑K) (n : Nat), Membership.mem K (P N n)
      hP : ∀ (N : ↑K) (n : Nat), (fun M N => Nonempty (L.Embedding ↑M ↑N)) (↑N) (P N …
      hFP : ∀ (N : ↑K) (n : Nat), (fun M N => Nonempty (L.Embedding ↑M ↑N)) (F (HAdd …
      G : Nat → ↑K := Nat.rec ⟨(F 0).out, ⋯⟩ fun n N => ⟨P N n, ⋯⟩
      f : (i j : Nat) → LE.le i j → L.Embedding ↑↑(G i) ↑↑(G j) := FirstOrder.Langua …
      this : DirectedSystem (fun n => ↑↑(G n)) fun i j h => ⇑(f i j h)
      ⊢ FirstOrder.Language.Structure.CG L ↑(CategoryTheory.Bundled.of (FirstOrder.L …
    -/
  · exact DirectLimit.cg _ (fun n => (fg _ (G n).2).cg)
    /-
      🎉 no goals
    -/
  · refine (age_directLimit (fun n ↦ (G n).val) f).trans
      (subset_antisymm (iUnion_subset fun n N hN => hp (G n).val (G n).2 hN) fun N KN => ?_)
    /-
      case intro.refine_2
      L : FirstOrder.Language
      K : Set (CategoryTheory.Bundled L.Structure)
      hn : K.Nonempty
      hc : (Set.image Quotient.mk' K).Countable
      fg : ∀ (M : CategoryTheory.Bundled L.Structure), Membership.mem K M → FirstOrd …
      hp : FirstOrder.Language.Hereditary K
      jep : FirstOrder.Language.JointEmbedding K
      F : Nat → Quotient FirstOrder.Language.equivSetoid
      hF : ∀ (a : CategoryTheory.Bundled L.Structure), Iff (Exists fun x => And (Mem …
      hF' : ∀ (n : Nat), Membership.mem K (F n).out
      P : ↑K → Nat → CategoryTheory.Bundled L.Structure
      hPK : ∀ (N : ↑K) (n : Nat), Membership.mem K (P N n)
      hP : ∀ (N : ↑K) (n : Nat), (fun M N => Nonempty (L.Embedding ↑M ↑N)) (↑N) (P N …
      hFP : ∀ (N : ↑K) (n : Nat), (fun M N => Nonempty (L.Embedding ↑M ↑N)) (F (HAdd …
      G : Nat → ↑K := Nat.rec ⟨(F 0).out, ⋯⟩ fun n N => ⟨P N n, ⋯⟩
      f : (i j : Nat) → LE.le i j → L.Embedding ↑↑(G i) ↑↑(G j) := FirstOrder.Langua …
      this : DirectedSystem (fun n => ↑↑(G n)) fun i j h => ⇑(f i j h)
      N : CategoryTheory.Bundled L.Structure
      KN : Membership.mem K N
      ⊢ Membership.mem (Set.iUnion fun i => L.age ↑↑(G i)) N
    -/
    have : Quotient.out (Quotient.mk' N) ≈ N := Quotient.eq_mk_iff_out.mp rfl
    /-
      case intro.refine_2
      L : FirstOrder.Language
      K : Set (CategoryTheory.Bundled L.Structure)
      hn : K.Nonempty
      hc : (Set.image Quotient.mk' K).Countable
      fg : ∀ (M : CategoryTheory.Bundled L.Structure), Membership.mem K M → FirstOrd …
      hp : FirstOrder.Language.Hereditary K
      jep : FirstOrder.Language.JointEmbedding K
      F : Nat → Quotient FirstOrder.Language.equivSetoid
      hF : ∀ (a : CategoryTheory.Bundled L.Structure), Iff (Exists fun x => And (Mem …
      hF' : ∀ (n : Nat), Membership.mem K (F n).out
      P : ↑K → Nat → CategoryTheory.Bundled L.Structure
      hPK : ∀ (N : ↑K) (n : Nat), Membership.mem K (P N n)
      hP : ∀ (N : ↑K) (n : Nat), (fun M N => Nonempty (L.Embedding ↑M ↑N)) (↑N) (P N …
      hFP : ∀ (N : ↑K) (n : Nat), (fun M N => Nonempty (L.Embedding ↑M ↑N)) (F (HAdd …
      G : Nat → ↑K := Nat.rec ⟨(F 0).out, ⋯⟩ fun n N => ⟨P N n, ⋯⟩
      f : (i j : Nat) → LE.le i j → L.Embedding ↑↑(G i) ↑↑(G j) := FirstOrder.Langua …
      this✝ : DirectedSystem (fun n => ↑↑(G n)) fun i j h => ⇑(f i j h)
      N : CategoryTheory.Bundled L.Structure
      KN : Membership.mem K N
      this : HasEquiv.Equiv (Quotient.mk' N).out N
      ⊢ Membership.mem (Set.iUnion fun i => L.age ↑↑(G i)) N
    -/
    obtain ⟨n, ⟨e⟩⟩ := (hF N).1 ⟨N, KN, this⟩
    /-
      case intro.refine_2.intro.intro
      L : FirstOrder.Language
      K : Set (CategoryTheory.Bundled L.Structure)
      hn : K.Nonempty
      hc : (Set.image Quotient.mk' K).Countable
      fg : ∀ (M : CategoryTheory.Bundled L.Structure), Membership.mem K M → FirstOrd …
      hp : FirstOrder.Language.Hereditary K
      jep : FirstOrder.Language.JointEmbedding K
      F : Nat → Quotient FirstOrder.Language.equivSetoid
      hF : ∀ (a : CategoryTheory.Bundled L.Structure), Iff (Exists fun x => And (Mem …
      hF' : ∀ (n : Nat), Membership.mem K (F n).out
      P : ↑K → Nat → CategoryTheory.Bundled L.Structure
      hPK : ∀ (N : ↑K) (n : Nat), Membership.mem K (P N n)
      hP : ∀ (N : ↑K) (n : Nat), (fun M N => Nonempty (L.Embedding ↑M ↑N)) (↑N) (P N …
      hFP : ∀ (N : ↑K) (n : Nat), (fun M N => Nonempty (L.Embedding ↑M ↑N)) (F (HAdd …
      G : Nat → ↑K := Nat.rec ⟨(F 0).out, ⋯⟩ fun n N => ⟨P N n, ⋯⟩
      f : (i j : Nat) → LE.le i j → L.Embedding ↑↑(G i) ↑↑(G j) := FirstOrder.Langua …
      this✝ : DirectedSystem (fun n => ↑↑(G n)) fun i j h => ⇑(f i j h)
      N : CategoryTheory.Bundled L.Structure
      KN : Membership.mem K N
      this : HasEquiv.Equiv (Quotient.mk' N).out N
      n : Nat
      e : L.Equiv ↑(F n).out ↑N
      ⊢ Membership.mem (Set.iUnion fun i => L.age ↑↑(G i)) N
    -/
    refine mem_iUnion_of_mem n ⟨fg _ KN, ⟨Embedding.comp ?_ e.symm.toEmbedding⟩⟩
    /-
      case intro.refine_2.intro.intro
      L : FirstOrder.Language
      K : Set (CategoryTheory.Bundled L.Structure)
      hn : K.Nonempty
      hc : (Set.image Quotient.mk' K).Countable
      fg : ∀ (M : CategoryTheory.Bundled L.Structure), Membership.mem K M → FirstOrd …
      hp : FirstOrder.Language.Hereditary K
      jep : FirstOrder.Language.JointEmbedding K
      F : Nat → Quotient FirstOrder.Language.equivSetoid
      hF : ∀ (a : CategoryTheory.Bundled L.Structure), Iff (Exists fun x => And (Mem …
      hF' : ∀ (n : Nat), Membership.mem K (F n).out
      P : ↑K → Nat → CategoryTheory.Bundled L.Structure
      hPK : ∀ (N : ↑K) (n : Nat), Membership.mem K (P N n)
      hP : ∀ (N : ↑K) (n : Nat), (fun M N => Nonempty (L.Embedding ↑M ↑N)) (↑N) (P N …
      hFP : ∀ (N : ↑K) (n : Nat), (fun M N => Nonempty (L.Embedding ↑M ↑N)) (F (HAdd …
      G : Nat → ↑K := Nat.rec ⟨(F 0).out, ⋯⟩ fun n N => ⟨P N n, ⋯⟩
      f : (i j : Nat) → LE.le i j → L.Embedding ↑↑(G i) ↑↑(G j) := FirstOrder.Langua …
      this✝ : DirectedSystem (fun n => ↑↑(G n)) fun i j h => ⇑(f i j h)
      N : CategoryTheory.Bundled L.Structure
      KN : Membership.mem K N
      this : HasEquiv.Equiv (Quotient.mk' N).out N
      n : Nat
      e : L.Equiv ↑(F n).out ↑N
      ⊢ L.Embedding ↑(F n).out ↑↑(G n)
    -/
    cases' n with n
      /-
        case intro.refine_2.intro.intro.zero
        L : FirstOrder.Language
        K : Set (CategoryTheory.Bundled L.Structure)
        hn : K.Nonempty
        hc : (Set.image Quotient.mk' K).Countable
        fg : ∀ (M : CategoryTheory.Bundled L.Structure), Membership.mem K M → FirstOrd …
        hp : FirstOrder.Language.Hereditary K
        jep : FirstOrder.Language.JointEmbedding K
        F : Nat → Quotient FirstOrder.Language.equivSetoid
        hF : ∀ (a : CategoryTheory.Bundled L.Structure), Iff (Exists fun x => And (Mem …
        hF' : ∀ (n : Nat), Membership.mem K (F n).out
        P : ↑K → Nat → CategoryTheory.Bundled L.Structure
        hPK : ∀ (N : ↑K) (n : Nat), Membership.mem K (P N n)
        hP : ∀ (N : ↑K) (n : Nat), (fun M N => Nonempty (L.Embedding ↑M ↑N)) (↑N) (P N …
        hFP : ∀ (N : ↑K) (n : Nat), (fun M N => Nonempty (L.Embedding ↑M ↑N)) (F (HAdd …
        G : Nat → ↑K := Nat.rec ⟨(F 0).out, ⋯⟩ fun n N => ⟨P N n, ⋯⟩
        f : (i j : Nat) → LE.le i j → L.Embedding ↑↑(G i) ↑↑(G j) := FirstOrder.Langua …
        this✝ : DirectedSystem (fun n => ↑↑(G n)) fun i j h => ⇑(f i j h)
        N : CategoryTheory.Bundled L.Structure
        KN : Membership.mem K N
        this : HasEquiv.Equiv (Quotient.mk' N).out N
        e : L.Equiv ↑(F 0).out ↑N
        ⊢ L.Embedding ↑(F 0).out ↑↑(G 0)
      -/
    · dsimp [G]; exact Embedding.refl _ _
                 /-
                   🎉 no goals
                 -/
      /-
        case intro.refine_2.intro.intro.succ
        L : FirstOrder.Language
        K : Set (CategoryTheory.Bundled L.Structure)
        hn : K.Nonempty
        hc : (Set.image Quotient.mk' K).Countable
        fg : ∀ (M : CategoryTheory.Bundled L.Structure), Membership.mem K M → FirstOrd …
        hp : FirstOrder.Language.Hereditary K
        jep : FirstOrder.Language.JointEmbedding K
        F : Nat → Quotient FirstOrder.Language.equivSetoid
        hF : ∀ (a : CategoryTheory.Bundled L.Structure), Iff (Exists fun x => And (Mem …
        hF' : ∀ (n : Nat), Membership.mem K (F n).out
        P : ↑K → Nat → CategoryTheory.Bundled L.Structure
        hPK : ∀ (N : ↑K) (n : Nat), Membership.mem K (P N n)
        hP : ∀ (N : ↑K) (n : Nat), (fun M N => Nonempty (L.Embedding ↑M ↑N)) (↑N) (P N …
        hFP : ∀ (N : ↑K) (n : Nat), (fun M N => Nonempty (L.Embedding ↑M ↑N)) (F (HAdd …
        G : Nat → ↑K := Nat.rec ⟨(F 0).out, ⋯⟩ fun n N => ⟨P N n, ⋯⟩
        f : (i j : Nat) → LE.le i j → L.Embedding ↑↑(G i) ↑↑(G j) := FirstOrder.Langua …
        this✝ : DirectedSystem (fun n => ↑↑(G n)) fun i j h => ⇑(f i j h)
        N : CategoryTheory.Bundled L.Structure
        KN : Membership.mem K N
        this : HasEquiv.Equiv (Quotient.mk' N).out N
        n : Nat
        e : L.Equiv ↑(F (HAdd.hAdd n 1)).out ↑N
        ⊢ L.Embedding ↑(F (HAdd.hAdd n 1)).out ↑↑(G (HAdd.hAdd n 1))
      -/
    · dsimp [G]; exact (hFP _ n).some
                 /-
                   🎉 no goals
                 -/


theorem exists_countable_is_age_of_iff [Countable (Σ l, L.Functions l)] :
    (∃ M : Bundled.{w} L.Structure, Countable M ∧ L.age M = K) ↔
      K.Nonempty ∧ (∀ M N : Bundled.{w} L.Structure, Nonempty (M ≃[L] N) → (M ∈ K ↔ N ∈ K)) ∧
      (Quotient.mk' '' K).Countable ∧ (∀ M : Bundled.{w} L.Structure, M ∈ K → Structure.FG L M) ∧
      Hereditary K ∧ JointEmbedding K := by
  /-
    L : FirstOrder.Language
    K : Set (CategoryTheory.Bundled L.Structure)
    inst✝ : Countable (Sigma fun l => L.Functions l)
    ⊢ Iff (Exists fun M => And (Countable ↑M) (Eq (L.age ↑M) K)) (And K.Nonempty ( …
  -/
  constructor
    /-
      case mp
      L : FirstOrder.Language
      K : Set (CategoryTheory.Bundled L.Structure)
      inst✝ : Countable (Sigma fun l => L.Functions l)
      ⊢ (Exists fun M => And (Countable ↑M) (Eq (L.age ↑M) K)) → And K.Nonempty (And …
    -/
  · rintro ⟨M, h1, h2, rfl⟩
    refine ⟨age.nonempty M, age.is_equiv_invariant L M, age.countable_quotient M, fun N hN => hN.1,
      age.hereditary M, age.jointEmbedding M⟩
    /-
      case mpr
      L : FirstOrder.Language
      K : Set (CategoryTheory.Bundled L.Structure)
      inst✝ : Countable (Sigma fun l => L.Functions l)
      ⊢ And K.Nonempty (And (∀ (M N : CategoryTheory.Bundled L.Structure), Nonempty  …
    -/
  · rintro ⟨Kn, _, cq, hfg, hp, jep⟩
    /-
      case mpr.intro.intro.intro.intro.intro
      L : FirstOrder.Language
      K : Set (CategoryTheory.Bundled L.Structure)
      inst✝ : Countable (Sigma fun l => L.Functions l)
      Kn : K.Nonempty
      left✝ : ∀ (M N : CategoryTheory.Bundled L.Structure), Nonempty (L.Equiv ↑M ↑N) …
      cq : (Set.image Quotient.mk' K).Countable
      hfg : ∀ (M : CategoryTheory.Bundled L.Structure), Membership.mem K M → FirstOr …
      hp : FirstOrder.Language.Hereditary K
      jep : FirstOrder.Language.JointEmbedding K
      ⊢ Exists fun M => And (Countable ↑M) (Eq (L.age ↑M) K)
    -/
    obtain ⟨M, hM, rfl⟩ := exists_cg_is_age_of Kn cq hfg hp jep
    /-
      case mpr.intro.intro.intro.intro.intro.intro.intro
      L : FirstOrder.Language
      inst✝ : Countable (Sigma fun l => L.Functions l)
      M : CategoryTheory.Bundled L.Structure
      hM : FirstOrder.Language.Structure.CG L ↑M
      Kn : (L.age ↑M).Nonempty
      left✝ : ∀ (M_1 N : CategoryTheory.Bundled L.Structure), Nonempty (L.Equiv ↑M_1 …
      cq : (Set.image Quotient.mk' (L.age ↑M)).Countable
      hfg : ∀ (M_1 : CategoryTheory.Bundled L.Structure), Membership.mem (L.age ↑M)  …
      hp : FirstOrder.Language.Hereditary (L.age ↑M)
      jep : FirstOrder.Language.JointEmbedding (L.age ↑M)
      ⊢ Exists fun M_1 => And (Countable ↑M_1) (Eq (L.age ↑M_1) (L.age ↑M))
    -/
    exact ⟨M, Structure.cg_iff_countable.1 hM, rfl⟩
    /-
      🎉 no goals
    -/


/-- A structure `M` is ultrahomogeneous if every embedding of a finitely generated substructure
into `M` extends to an automorphism of `M`. -/
def IsUltrahomogeneous : Prop :=
  ∀ (S : L.Substructure M) (_ : S.FG) (f : S ↪[L] M),
    ∃ g : M ≃[L] M, f = g.toEmbedding.comp S.subtype


/-- A structure `M` is a Fraïssé limit for a class `K` if it is countably generated,
ultrahomogeneous, and has age `K`. -/
structure IsFraisseLimit [Countable (Σ l, L.Functions l)] [Countable M] : Prop where
  protected ultrahomogeneous : IsUltrahomogeneous L M
  protected age : L.age M = K


/-- Any embedding from a finitely generated `S` to an ultrahomogeneous structure `M`
can be extended to an embedding from any structure with an embedding to `M`. -/
theorem IsUltrahomogeneous.extend_embedding (M_homog : L.IsUltrahomogeneous M) {S : Type*}
    [L.Structure S] (S_FG : FG L S) {T : Type*} [L.Structure T] [h : Nonempty (T ↪[L] M)]
    (f : S ↪[L] M) (g : S ↪[L] T) :
    ∃ f' : T ↪[L] M, f = f'.comp g := by
  /-
    L : FirstOrder.Language
    M : Type w
    inst✝² : L.Structure M
    M_homog : L.IsUltrahomogeneous M
    S : Type u_1
    inst✝¹ : L.Structure S
    S_FG : FirstOrder.Language.Structure.FG L S
    T : Type u_2
    inst✝ : L.Structure T
    h : Nonempty (L.Embedding T M)
    f : L.Embedding S M
    g : L.Embedding S T
    ⊢ Exists fun f' => Eq f (f'.comp g)
  -/
  let ⟨r⟩ := h
  /-
    L : FirstOrder.Language
    M : Type w
    inst✝² : L.Structure M
    M_homog : L.IsUltrahomogeneous M
    S : Type u_1
    inst✝¹ : L.Structure S
    S_FG : FirstOrder.Language.Structure.FG L S
    T : Type u_2
    inst✝ : L.Structure T
    h : Nonempty (L.Embedding T M)
    f : L.Embedding S M
    g : L.Embedding S T
    r : L.Embedding T M
    ⊢ Exists fun f' => Eq f (f'.comp g)
  -/
  let s := r.comp g
  /-
    L : FirstOrder.Language
    M : Type w
    inst✝² : L.Structure M
    M_homog : L.IsUltrahomogeneous M
    S : Type u_1
    inst✝¹ : L.Structure S
    S_FG : FirstOrder.Language.Structure.FG L S
    T : Type u_2
    inst✝ : L.Structure T
    h : Nonempty (L.Embedding T M)
    f : L.Embedding S M
    g : L.Embedding S T
    r : L.Embedding T M
    s : L.Embedding S M := r.comp g
    ⊢ Exists fun f' => Eq f (f'.comp g)
  -/
  let ⟨t, eq⟩ := M_homog s.toHom.range (S_FG.range s.toHom) (f.comp s.equivRange.symm.toEmbedding)
  /-
    L : FirstOrder.Language
    M : Type w
    inst✝² : L.Structure M
    M_homog : L.IsUltrahomogeneous M
    S : Type u_1
    inst✝¹ : L.Structure S
    S_FG : FirstOrder.Language.Structure.FG L S
    T : Type u_2
    inst✝ : L.Structure T
    h : Nonempty (L.Embedding T M)
    f : L.Embedding S M
    g : L.Embedding S T
    r : L.Embedding T M
    s : L.Embedding S M := r.comp g
    t : L.Equiv M M
    eq : Eq (f.comp s.equivRange.symm.toEmbedding) (t.toEmbedding.comp s.toHom.ran …
    ⊢ Exists fun f' => Eq f (f'.comp g)
  -/
  use t.toEmbedding.comp r
  /-
    case h
    L : FirstOrder.Language
    M : Type w
    inst✝² : L.Structure M
    M_homog : L.IsUltrahomogeneous M
    S : Type u_1
    inst✝¹ : L.Structure S
    S_FG : FirstOrder.Language.Structure.FG L S
    T : Type u_2
    inst✝ : L.Structure T
    h : Nonempty (L.Embedding T M)
    f : L.Embedding S M
    g : L.Embedding S T
    r : L.Embedding T M
    s : L.Embedding S M := r.comp g
    t : L.Equiv M M
    eq : Eq (f.comp s.equivRange.symm.toEmbedding) (t.toEmbedding.comp s.toHom.ran …
    ⊢ Eq f ((t.toEmbedding.comp r).comp g)
  -/
  change _ = t.toEmbedding.comp s
  /-
    case h
    L : FirstOrder.Language
    M : Type w
    inst✝² : L.Structure M
    M_homog : L.IsUltrahomogeneous M
    S : Type u_1
    inst✝¹ : L.Structure S
    S_FG : FirstOrder.Language.Structure.FG L S
    T : Type u_2
    inst✝ : L.Structure T
    h : Nonempty (L.Embedding T M)
    f : L.Embedding S M
    g : L.Embedding S T
    r : L.Embedding T M
    s : L.Embedding S M := r.comp g
    t : L.Equiv M M
    eq : Eq (f.comp s.equivRange.symm.toEmbedding) (t.toEmbedding.comp s.toHom.ran …
    ⊢ Eq f (t.toEmbedding.comp s)
  -/
  ext x
  /-
    case h.h
    L : FirstOrder.Language
    M : Type w
    inst✝² : L.Structure M
    M_homog : L.IsUltrahomogeneous M
    S : Type u_1
    inst✝¹ : L.Structure S
    S_FG : FirstOrder.Language.Structure.FG L S
    T : Type u_2
    inst✝ : L.Structure T
    h : Nonempty (L.Embedding T M)
    f : L.Embedding S M
    g : L.Embedding S T
    r : L.Embedding T M
    s : L.Embedding S M := r.comp g
    t : L.Equiv M M
    eq : Eq (f.comp s.equivRange.symm.toEmbedding) (t.toEmbedding.comp s.toHom.ran …
    x : S
    ⊢ Eq (f x) ((t.toEmbedding.comp s) x)
  -/
  have eq' := congr_fun (congr_arg DFunLike.coe eq) ⟨s x, Hom.mem_range.2 ⟨x, rfl⟩⟩
  simp only [Embedding.comp_apply, Hom.comp_apply,
    Equiv.coe_toHom, Embedding.coe_toHom, coeSubtype] at eq'
  /-
    case h.h
    L : FirstOrder.Language
    M : Type w
    inst✝² : L.Structure M
    M_homog : L.IsUltrahomogeneous M
    S : Type u_1
    inst✝¹ : L.Structure S
    S_FG : FirstOrder.Language.Structure.FG L S
    T : Type u_2
    inst✝ : L.Structure T
    h : Nonempty (L.Embedding T M)
    f : L.Embedding S M
    g : L.Embedding S T
    r : L.Embedding T M
    s : L.Embedding S M := r.comp g
    t : L.Equiv M M
    eq : Eq (f.comp s.equivRange.symm.toEmbedding) (t.toEmbedding.comp s.toHom.ran …
    x : S
    eq' : Eq (f (s.equivRange.symm.toEmbedding ⟨s x, ⋯⟩)) (t.toEmbedding (s x))
    ⊢ Eq (f x) ((t.toEmbedding.comp s) x)
  -/
  simp only [Embedding.comp_apply, ← eq', Equiv.coe_toEmbedding, EmbeddingLike.apply_eq_iff_eq]
  /-
    case h.h
    L : FirstOrder.Language
    M : Type w
    inst✝² : L.Structure M
    M_homog : L.IsUltrahomogeneous M
    S : Type u_1
    inst✝¹ : L.Structure S
    S_FG : FirstOrder.Language.Structure.FG L S
    T : Type u_2
    inst✝ : L.Structure T
    h : Nonempty (L.Embedding T M)
    f : L.Embedding S M
    g : L.Embedding S T
    r : L.Embedding T M
    s : L.Embedding S M := r.comp g
    t : L.Equiv M M
    eq : Eq (f.comp s.equivRange.symm.toEmbedding) (t.toEmbedding.comp s.toHom.ran …
    x : S
    eq' : Eq (f (s.equivRange.symm.toEmbedding ⟨s x, ⋯⟩)) (t.toEmbedding (s x))
    ⊢ Eq x (s.equivRange.symm ⟨s x, ⋯⟩)
  -/
  apply (Embedding.equivRange (Embedding.comp r g)).injective
  /-
    case h.h.a
    L : FirstOrder.Language
    M : Type w
    inst✝² : L.Structure M
    M_homog : L.IsUltrahomogeneous M
    S : Type u_1
    inst✝¹ : L.Structure S
    S_FG : FirstOrder.Language.Structure.FG L S
    T : Type u_2
    inst✝ : L.Structure T
    h : Nonempty (L.Embedding T M)
    f : L.Embedding S M
    g : L.Embedding S T
    r : L.Embedding T M
    s : L.Embedding S M := r.comp g
    t : L.Equiv M M
    eq : Eq (f.comp s.equivRange.symm.toEmbedding) (t.toEmbedding.comp s.toHom.ran …
    x : S
    eq' : Eq (f (s.equivRange.symm.toEmbedding ⟨s x, ⋯⟩)) (t.toEmbedding (s x))
    ⊢ Eq ((r.comp g).equivRange x) ((r.comp g).equivRange (s.equivRange.symm ⟨s x, …
  -/
  ext
  /-
    case h.h.a.a
    L : FirstOrder.Language
    M : Type w
    inst✝² : L.Structure M
    M_homog : L.IsUltrahomogeneous M
    S : Type u_1
    inst✝¹ : L.Structure S
    S_FG : FirstOrder.Language.Structure.FG L S
    T : Type u_2
    inst✝ : L.Structure T
    h : Nonempty (L.Embedding T M)
    f : L.Embedding S M
    g : L.Embedding S T
    r : L.Embedding T M
    s : L.Embedding S M := r.comp g
    t : L.Equiv M M
    eq : Eq (f.comp s.equivRange.symm.toEmbedding) (t.toEmbedding.comp s.toHom.ran …
    x : S
    eq' : Eq (f (s.equivRange.symm.toEmbedding ⟨s x, ⋯⟩)) (t.toEmbedding (s x))
    ⊢ Eq ↑((r.comp g).equivRange x) ↑((r.comp g).equivRange (s.equivRange.symm ⟨s  …
  -/
  simp only [Equiv.apply_symm_apply, Embedding.equivRange_apply, s]
  /-
    🎉 no goals
  -/


/-- A countably generated structure is ultrahomogeneous if and only if any equivalence between
finitely generated substructures can be extended to any element in the domain.-/
theorem isUltrahomogeneous_iff_IsExtensionPair (M_CG : CG L M) : L.IsUltrahomogeneous M ↔
    L.IsExtensionPair M M := by
  /-
    L : FirstOrder.Language
    M : Type w
    inst✝ : L.Structure M
    M_CG : FirstOrder.Language.Structure.CG L M
    ⊢ Iff (L.IsUltrahomogeneous M) (L.IsExtensionPair M M)
  -/
  constructor
    /-
      case mp
      L : FirstOrder.Language
      M : Type w
      inst✝ : L.Structure M
      M_CG : FirstOrder.Language.Structure.CG L M
      ⊢ L.IsUltrahomogeneous M → L.IsExtensionPair M M
    -/
  · intro M_homog ⟨f, f_FG⟩ m
    /-
      case mp
      L : FirstOrder.Language
      M : Type w
      inst✝ : L.Structure M
      M_CG : FirstOrder.Language.Structure.CG L M
      M_homog : L.IsUltrahomogeneous M
      f : L.PartialEquiv M M
      f_FG : f.dom.FG
      m : M
      ⊢ Exists fun g => And (Membership.mem (↑g).dom m) (LE.le ⟨f, f_FG⟩ g)
    -/
    let S := f.dom ⊔ closure L {m}
    /-
      case mp
      L : FirstOrder.Language
      M : Type w
      inst✝ : L.Structure M
      M_CG : FirstOrder.Language.Structure.CG L M
      M_homog : L.IsUltrahomogeneous M
      f : L.PartialEquiv M M
      f_FG : f.dom.FG
      m : M
      S : L.Substructure M := Max.max f.dom ((FirstOrder.Language.Substructure.closu …
      ⊢ Exists fun g => And (Membership.mem (↑g).dom m) (LE.le ⟨f, f_FG⟩ g)
    -/
    have dom_le_S : f.dom ≤ S := le_sup_left
    let ⟨f', eq_f'⟩ := M_homog.extend_embedding (f.dom.fg_iff_structure_fg.1 f_FG)
      ((subtype _).comp f.toEquiv.toEmbedding) (inclusion dom_le_S) (h := ⟨subtype _⟩)
    refine ⟨⟨⟨S, f'.toHom.range, f'.equivRange⟩, f_FG.sup (fg_closure_singleton _)⟩,
      subset_closure.trans (le_sup_right : _ ≤ S) (mem_singleton m), ⟨dom_le_S, ?_⟩⟩
    /-
      case mp
      L : FirstOrder.Language
      M : Type w
      inst✝ : L.Structure M
      M_CG : FirstOrder.Language.Structure.CG L M
      M_homog : L.IsUltrahomogeneous M
      f : L.PartialEquiv M M
      f_FG : f.dom.FG
      m : M
      S : L.Substructure M := Max.max f.dom ((FirstOrder.Language.Substructure.closu …
      dom_le_S : LE.le f.dom S
      f' : L.Embedding (Subtype fun x => Membership.mem S x) M
      eq_f' : Eq (f.cod.subtype.comp f.toEquiv.toEmbedding) (f'.comp (FirstOrder.Lan …
      ⊢ Eq ((↑⟨{ dom := S, cod := f'.toHom.range, toEquiv := f'.equivRange }, ⋯⟩).co …
    -/
    ext
    simp only [Embedding.comp_apply, Equiv.coe_toEmbedding, coeSubtype, eq_f',
      Embedding.equivRange_apply, Substructure.coe_inclusion, EmbeddingLike.apply_eq_iff_eq]
    /-
      case mpr
      L : FirstOrder.Language
      M : Type w
      inst✝ : L.Structure M
      M_CG : FirstOrder.Language.Structure.CG L M
      ⊢ L.IsExtensionPair M M → L.IsUltrahomogeneous M
    -/
  · intro h S S_FG f
    let ⟨g, ⟨dom_le_dom, eq⟩⟩ :=
      equiv_between_cg M_CG M_CG ⟨⟨S, f.toHom.range, f.equivRange⟩, S_FG⟩ h h
    /-
      case mpr
      L : FirstOrder.Language
      M : Type w
      inst✝ : L.Structure M
      M_CG : FirstOrder.Language.Structure.CG L M
      h : L.IsExtensionPair M M
      S : L.Substructure M
      S_FG : S.FG
      f : L.Embedding (Subtype fun x => Membership.mem S x) M
      g : L.Equiv M M
      dom_le_dom : LE.le (↑⟨{ dom := S, cod := f.toHom.range, toEquiv := f.equivRang …
      eq : Eq (g.toEmbedding.toPartialEquiv.cod.subtype.comp (g.toEmbedding.toPartia …
      ⊢ Exists fun g => Eq f (g.toEmbedding.comp S.subtype)
    -/
    use g
    /-
      case h
      L : FirstOrder.Language
      M : Type w
      inst✝ : L.Structure M
      M_CG : FirstOrder.Language.Structure.CG L M
      h : L.IsExtensionPair M M
      S : L.Substructure M
      S_FG : S.FG
      f : L.Embedding (Subtype fun x => Membership.mem S x) M
      g : L.Equiv M M
      dom_le_dom : LE.le (↑⟨{ dom := S, cod := f.toHom.range, toEquiv := f.equivRang …
      eq : Eq (g.toEmbedding.toPartialEquiv.cod.subtype.comp (g.toEmbedding.toPartia …
      ⊢ Eq f (g.toEmbedding.comp S.subtype)
    -/
    simp only [Embedding.subtype_equivRange] at eq
    /-
      case h
      L : FirstOrder.Language
      M : Type w
      inst✝ : L.Structure M
      M_CG : FirstOrder.Language.Structure.CG L M
      h : L.IsExtensionPair M M
      S : L.Substructure M
      S_FG : S.FG
      f : L.Embedding (Subtype fun x => Membership.mem S x) M
      g : L.Equiv M M
      dom_le_dom : LE.le (↑⟨{ dom := S, cod := f.toHom.range, toEquiv := f.equivRang …
      eq : Eq (g.toEmbedding.toPartialEquiv.cod.subtype.comp (g.toEmbedding.toPartia …
      ⊢ Eq f (g.toEmbedding.comp S.subtype)
    -/
    rw [← eq]
    /-
      case h
      L : FirstOrder.Language
      M : Type w
      inst✝ : L.Structure M
      M_CG : FirstOrder.Language.Structure.CG L M
      h : L.IsExtensionPair M M
      S : L.Substructure M
      S_FG : S.FG
      f : L.Embedding (Subtype fun x => Membership.mem S x) M
      g : L.Equiv M M
      dom_le_dom : LE.le (↑⟨{ dom := S, cod := f.toHom.range, toEquiv := f.equivRang …
      eq : Eq (g.toEmbedding.toPartialEquiv.cod.subtype.comp (g.toEmbedding.toPartia …
      ⊢ Eq (g.toEmbedding.toPartialEquiv.cod.subtype.comp (g.toEmbedding.toPartialEq …
    -/
    ext
    /-
      case h.h
      L : FirstOrder.Language
      M : Type w
      inst✝ : L.Structure M
      M_CG : FirstOrder.Language.Structure.CG L M
      h : L.IsExtensionPair M M
      S : L.Substructure M
      S_FG : S.FG
      f : L.Embedding (Subtype fun x => Membership.mem S x) M
      g : L.Equiv M M
      dom_le_dom : LE.le (↑⟨{ dom := S, cod := f.toHom.range, toEquiv := f.equivRang …
      eq : Eq (g.toEmbedding.toPartialEquiv.cod.subtype.comp (g.toEmbedding.toPartia …
      x✝ : Subtype fun x => Membership.mem S x
      ⊢ Eq ((g.toEmbedding.toPartialEquiv.cod.subtype.comp (g.toEmbedding.toPartialE …
    -/
    rfl
    /-
      🎉 no goals
    -/


theorem IsUltrahomogeneous.amalgamation_age (h : L.IsUltrahomogeneous M) :
    Amalgamation (L.age M) := by
  /-
    L : FirstOrder.Language
    M : Type w
    inst✝ : L.Structure M
    h : L.IsUltrahomogeneous M
    ⊢ FirstOrder.Language.Amalgamation (L.age M)
  -/
  rintro N P Q NP NQ ⟨Nfg, ⟨-⟩⟩ ⟨Pfg, ⟨PM⟩⟩ ⟨Qfg, ⟨QM⟩⟩
  obtain ⟨g, hg⟩ := h (PM.comp NP).toHom.range (Nfg.range _)
    ((QM.comp NQ).comp (PM.comp NP).equivRange.symm.toEmbedding)
  /-
    case intro.intro.intro.intro.intro.intro.intro
    L : FirstOrder.Language
    M : Type w
    inst✝ : L.Structure M
    h : L.IsUltrahomogeneous M
    N P Q : CategoryTheory.Bundled L.Structure
    NP : L.Embedding ↑N ↑P
    NQ : L.Embedding ↑N ↑Q
    Nfg : FirstOrder.Language.Structure.FG L ↑N
    Pfg : FirstOrder.Language.Structure.FG L ↑P
    PM : L.Embedding (↑P) M
    Qfg : FirstOrder.Language.Structure.FG L ↑Q
    QM : L.Embedding (↑Q) M
    g : L.Equiv M M
    hg : Eq ((QM.comp NQ).comp (PM.comp NP).equivRange.symm.toEmbedding) (g.toEmbe …
    ⊢ Exists fun Q_1 => Exists fun NQ_1 => Exists fun PQ => And (Membership.mem (L …
  -/
  let s := (g.toHom.comp PM.toHom).range ⊔ QM.toHom.range
  refine ⟨Bundled.of s,
    Embedding.comp (Substructure.inclusion le_sup_left)
      (g.toEmbedding.comp PM).equivRange.toEmbedding,
    Embedding.comp (Substructure.inclusion le_sup_right) QM.equivRange.toEmbedding,
    ⟨(fg_iff_structure_fg _).1 (FG.sup (Pfg.range _) (Qfg.range _)), ⟨Substructure.subtype _⟩⟩, ?_⟩
  /-
    case intro.intro.intro.intro.intro.intro.intro
    L : FirstOrder.Language
    M : Type w
    inst✝ : L.Structure M
    h : L.IsUltrahomogeneous M
    N P Q : CategoryTheory.Bundled L.Structure
    NP : L.Embedding ↑N ↑P
    NQ : L.Embedding ↑N ↑Q
    Nfg : FirstOrder.Language.Structure.FG L ↑N
    Pfg : FirstOrder.Language.Structure.FG L ↑P
    PM : L.Embedding (↑P) M
    Qfg : FirstOrder.Language.Structure.FG L ↑Q
    QM : L.Embedding (↑Q) M
    g : L.Equiv M M
    hg : Eq ((QM.comp NQ).comp (PM.comp NP).equivRange.symm.toEmbedding) (g.toEmbe …
    s : L.Substructure M := Max.max (g.toHom.comp PM.toHom).range QM.toHom.range
    ⊢ Eq (((FirstOrder.Language.Substructure.inclusion ⋯).comp (g.toEmbedding.comp …
  -/
  ext n
  /-
    case intro.intro.intro.intro.intro.intro.intro.h
    L : FirstOrder.Language
    M : Type w
    inst✝ : L.Structure M
    h : L.IsUltrahomogeneous M
    N P Q : CategoryTheory.Bundled L.Structure
    NP : L.Embedding ↑N ↑P
    NQ : L.Embedding ↑N ↑Q
    Nfg : FirstOrder.Language.Structure.FG L ↑N
    Pfg : FirstOrder.Language.Structure.FG L ↑P
    PM : L.Embedding (↑P) M
    Qfg : FirstOrder.Language.Structure.FG L ↑Q
    QM : L.Embedding (↑Q) M
    g : L.Equiv M M
    hg : Eq ((QM.comp NQ).comp (PM.comp NP).equivRange.symm.toEmbedding) (g.toEmbe …
    s : L.Substructure M := Max.max (g.toHom.comp PM.toHom).range QM.toHom.range
    n : ↑N
    ⊢ Eq ((((FirstOrder.Language.Substructure.inclusion ⋯).comp (g.toEmbedding.com …
  -/
  apply Subtype.ext
  /-
    case intro.intro.intro.intro.intro.intro.intro.h.a
    L : FirstOrder.Language
    M : Type w
    inst✝ : L.Structure M
    h : L.IsUltrahomogeneous M
    N P Q : CategoryTheory.Bundled L.Structure
    NP : L.Embedding ↑N ↑P
    NQ : L.Embedding ↑N ↑Q
    Nfg : FirstOrder.Language.Structure.FG L ↑N
    Pfg : FirstOrder.Language.Structure.FG L ↑P
    PM : L.Embedding (↑P) M
    Qfg : FirstOrder.Language.Structure.FG L ↑Q
    QM : L.Embedding (↑Q) M
    g : L.Equiv M M
    hg : Eq ((QM.comp NQ).comp (PM.comp NP).equivRange.symm.toEmbedding) (g.toEmbe …
    s : L.Substructure M := Max.max (g.toHom.comp PM.toHom).range QM.toHom.range
    n : ↑N
    ⊢ Eq ↑((((FirstOrder.Language.Substructure.inclusion ⋯).comp (g.toEmbedding.co …
  -/
  have hgn := (Embedding.ext_iff.1 hg) ((PM.comp NP).equivRange n)
  simp only [Embedding.comp_apply, Equiv.coe_toEmbedding, Equiv.symm_apply_apply,
    Substructure.coeSubtype, Embedding.equivRange_apply] at hgn
  /-
    case intro.intro.intro.intro.intro.intro.intro.h.a
    L : FirstOrder.Language
    M : Type w
    inst✝ : L.Structure M
    h : L.IsUltrahomogeneous M
    N P Q : CategoryTheory.Bundled L.Structure
    NP : L.Embedding ↑N ↑P
    NQ : L.Embedding ↑N ↑Q
    Nfg : FirstOrder.Language.Structure.FG L ↑N
    Pfg : FirstOrder.Language.Structure.FG L ↑P
    PM : L.Embedding (↑P) M
    Qfg : FirstOrder.Language.Structure.FG L ↑Q
    QM : L.Embedding (↑Q) M
    g : L.Equiv M M
    hg : Eq ((QM.comp NQ).comp (PM.comp NP).equivRange.symm.toEmbedding) (g.toEmbe …
    s : L.Substructure M := Max.max (g.toHom.comp PM.toHom).range QM.toHom.range
    n : ↑N
    hgn : Eq (QM (NQ n)) (g (PM (NP n)))
    ⊢ Eq ↑((((FirstOrder.Language.Substructure.inclusion ⋯).comp (g.toEmbedding.co …
  -/
  simp only [Embedding.comp_apply, Equiv.coe_toEmbedding]
  /-
    case intro.intro.intro.intro.intro.intro.intro.h.a
    L : FirstOrder.Language
    M : Type w
    inst✝ : L.Structure M
    h : L.IsUltrahomogeneous M
    N P Q : CategoryTheory.Bundled L.Structure
    NP : L.Embedding ↑N ↑P
    NQ : L.Embedding ↑N ↑Q
    Nfg : FirstOrder.Language.Structure.FG L ↑N
    Pfg : FirstOrder.Language.Structure.FG L ↑P
    PM : L.Embedding (↑P) M
    Qfg : FirstOrder.Language.Structure.FG L ↑Q
    QM : L.Embedding (↑Q) M
    g : L.Equiv M M
    hg : Eq ((QM.comp NQ).comp (PM.comp NP).equivRange.symm.toEmbedding) (g.toEmbe …
    s : L.Substructure M := Max.max (g.toHom.comp PM.toHom).range QM.toHom.range
    n : ↑N
    hgn : Eq (QM (NQ n)) (g (PM (NP n)))
    ⊢ Eq ↑((FirstOrder.Language.Substructure.inclusion ⋯) ((g.toEmbedding.comp PM) …
  -/
  erw [Substructure.coe_inclusion, Substructure.coe_inclusion]
  simp only [Embedding.comp_apply, Equiv.coe_toEmbedding, Set.coe_inclusion,
    Embedding.equivRange_apply, hgn]
  -- This used to be `simp only [...]` before https://github.com/leanprover/lean4/pull/2644
  erw [Embedding.comp_apply, Equiv.coe_toEmbedding,
    Embedding.equivRange_apply]
  /-
    case intro.intro.intro.intro.intro.intro.intro.h.a
    L : FirstOrder.Language
    M : Type w
    inst✝ : L.Structure M
    h : L.IsUltrahomogeneous M
    N P Q : CategoryTheory.Bundled L.Structure
    NP : L.Embedding ↑N ↑P
    NQ : L.Embedding ↑N ↑Q
    Nfg : FirstOrder.Language.Structure.FG L ↑N
    Pfg : FirstOrder.Language.Structure.FG L ↑P
    PM : L.Embedding (↑P) M
    Qfg : FirstOrder.Language.Structure.FG L ↑Q
    QM : L.Embedding (↑Q) M
    g : L.Equiv M M
    hg : Eq ((QM.comp NQ).comp (PM.comp NP).equivRange.symm.toEmbedding) (g.toEmbe …
    s : L.Substructure M := Max.max (g.toHom.comp PM.toHom).range QM.toHom.range
    n : ↑N
    hgn : Eq (QM (NQ n)) (g (PM (NP n)))
    ⊢ Eq ((g.toEmbedding.comp PM) (NP n)) (g (PM (NP n)))
  -/
  simp
  /-
    🎉 no goals
  -/


theorem IsUltrahomogeneous.age_isFraisse [Countable M] (h : L.IsUltrahomogeneous M) :
    IsFraisse (L.age M) :=
  ⟨age.nonempty M, fun _ hN => hN.1, age.countable_quotient M,
    age.hereditary M, age.jointEmbedding M, h.amalgamation_age⟩


/-- If a class has a Fraïssé limit, it must be Fraïssé. -/
theorem isFraisse [Countable (Σ l, L.Functions l)] [Countable M] (h : IsFraisseLimit K M) :
    IsFraisse K :=
  (congr rfl h.age).mp h.ultrahomogeneous.age_isFraisse


protected theorem isExtensionPair : L.IsExtensionPair M N := by
  /-
    L : FirstOrder.Language
    K : Set (CategoryTheory.Bundled L.Structure)
    M : Type w
    inst✝⁴ : L.Structure M
    N : Type w
    inst✝³ : L.Structure N
    inst✝² : Countable (Sigma fun l => L.Functions l)
    inst✝¹ : Countable M
    inst✝ : Countable N
    hM : FirstOrder.Language.IsFraisseLimit K M
    hN : FirstOrder.Language.IsFraisseLimit K N
    ⊢ L.IsExtensionPair M N
  -/
  intro ⟨f, f_FG⟩ m
  /-
    L : FirstOrder.Language
    K : Set (CategoryTheory.Bundled L.Structure)
    M : Type w
    inst✝⁴ : L.Structure M
    N : Type w
    inst✝³ : L.Structure N
    inst✝² : Countable (Sigma fun l => L.Functions l)
    inst✝¹ : Countable M
    inst✝ : Countable N
    hM : FirstOrder.Language.IsFraisseLimit K M
    hN : FirstOrder.Language.IsFraisseLimit K N
    f : L.PartialEquiv M N
    f_FG : f.dom.FG
    m : M
    ⊢ Exists fun g => And (Membership.mem (↑g).dom m) (LE.le ⟨f, f_FG⟩ g)
  -/
  let S := f.dom ⊔ closure L {m}
  /-
    L : FirstOrder.Language
    K : Set (CategoryTheory.Bundled L.Structure)
    M : Type w
    inst✝⁴ : L.Structure M
    N : Type w
    inst✝³ : L.Structure N
    inst✝² : Countable (Sigma fun l => L.Functions l)
    inst✝¹ : Countable M
    inst✝ : Countable N
    hM : FirstOrder.Language.IsFraisseLimit K M
    hN : FirstOrder.Language.IsFraisseLimit K N
    f : L.PartialEquiv M N
    f_FG : f.dom.FG
    m : M
    S : L.Substructure M := Max.max f.dom ((FirstOrder.Language.Substructure.closu …
    ⊢ Exists fun g => And (Membership.mem (↑g).dom m) (LE.le ⟨f, f_FG⟩ g)
  -/
  have S_FG : S.FG := f_FG.sup (Substructure.fg_closure_singleton _)
  have S_in_age_N : ⟨S, inferInstance⟩ ∈ L.age N := by
    rw [hN.age, ← hM.age]
    exact ⟨(fg_iff_structure_fg S).1 S_FG, ⟨subtype _⟩⟩
  /-
    L : FirstOrder.Language
    K : Set (CategoryTheory.Bundled L.Structure)
    M : Type w
    inst✝⁴ : L.Structure M
    N : Type w
    inst✝³ : L.Structure N
    inst✝² : Countable (Sigma fun l => L.Functions l)
    inst✝¹ : Countable M
    inst✝ : Countable N
    hM : FirstOrder.Language.IsFraisseLimit K M
    hN : FirstOrder.Language.IsFraisseLimit K N
    f : L.PartialEquiv M N
    f_FG : f.dom.FG
    m : M
    S : L.Substructure M := Max.max f.dom ((FirstOrder.Language.Substructure.closu …
    S_FG : S.FG
    S_in_age_N : Membership.mem (L.age N) { α := Subtype fun x => Membership.mem S …
    ⊢ Exists fun g => And (Membership.mem (↑g).dom m) (LE.le ⟨f, f_FG⟩ g)
  -/
  haveI nonempty_S_N : Nonempty (S ↪[L] N) := S_in_age_N.2
  let ⟨g, g_eq⟩ := hN.ultrahomogeneous.extend_embedding (f.dom.fg_iff_structure_fg.1 f_FG)
    ((subtype f.cod).comp f.toEquiv.toEmbedding) (inclusion (le_sup_left : _ ≤ S))
  refine ⟨⟨⟨S, g.toHom.range, g.equivRange⟩, S_FG⟩,
    subset_closure.trans (le_sup_right : _ ≤ S) (mem_singleton m), ⟨le_sup_left, ?_⟩⟩
  /-
    L : FirstOrder.Language
    K : Set (CategoryTheory.Bundled L.Structure)
    M : Type w
    inst✝⁴ : L.Structure M
    N : Type w
    inst✝³ : L.Structure N
    inst✝² : Countable (Sigma fun l => L.Functions l)
    inst✝¹ : Countable M
    inst✝ : Countable N
    hM : FirstOrder.Language.IsFraisseLimit K M
    hN : FirstOrder.Language.IsFraisseLimit K N
    f : L.PartialEquiv M N
    f_FG : f.dom.FG
    m : M
    S : L.Substructure M := Max.max f.dom ((FirstOrder.Language.Substructure.closu …
    S_FG : S.FG
    S_in_age_N : Membership.mem (L.age N) { α := Subtype fun x => Membership.mem S …
    nonempty_S_N : Nonempty (L.Embedding (Subtype fun x => Membership.mem S x) N)
    g : L.Embedding (Subtype fun x => Membership.mem (Max.max f.dom ((FirstOrder.L …
    g_eq : Eq (f.cod.subtype.comp f.toEquiv.toEmbedding) (g.comp (FirstOrder.Langu …
    ⊢ Eq ((↑⟨{ dom := S, cod := g.toHom.range, toEquiv := g.equivRange }, S_FG⟩).c …
  -/
  ext
  /-
    case h
    L : FirstOrder.Language
    K : Set (CategoryTheory.Bundled L.Structure)
    M : Type w
    inst✝⁴ : L.Structure M
    N : Type w
    inst✝³ : L.Structure N
    inst✝² : Countable (Sigma fun l => L.Functions l)
    inst✝¹ : Countable M
    inst✝ : Countable N
    hM : FirstOrder.Language.IsFraisseLimit K M
    hN : FirstOrder.Language.IsFraisseLimit K N
    f : L.PartialEquiv M N
    f_FG : f.dom.FG
    m : M
    S : L.Substructure M := Max.max f.dom ((FirstOrder.Language.Substructure.closu …
    S_FG : S.FG
    S_in_age_N : Membership.mem (L.age N) { α := Subtype fun x => Membership.mem S …
    nonempty_S_N : Nonempty (L.Embedding (Subtype fun x => Membership.mem S x) N)
    g : L.Embedding (Subtype fun x => Membership.mem (Max.max f.dom ((FirstOrder.L …
    g_eq : Eq (f.cod.subtype.comp f.toEquiv.toEmbedding) (g.comp (FirstOrder.Langu …
    x✝ : Subtype fun x => Membership.mem (↑⟨f, f_FG⟩).dom x
    ⊢ Eq (((↑⟨{ dom := S, cod := g.toHom.range, toEquiv := g.equivRange }, S_FG⟩). …
  -/
  simp [S, Subtype.mk_le_mk, PartialEquiv.le_def, g_eq]
  /-
    🎉 no goals
  -/


/-- The Fraïssé limit of a class is unique, in that any two Fraïssé limits are isomorphic. -/
theorem nonempty_equiv : Nonempty (M ≃[L] N) := by
  /-
    L : FirstOrder.Language
    K : Set (CategoryTheory.Bundled L.Structure)
    M : Type w
    inst✝⁴ : L.Structure M
    N : Type w
    inst✝³ : L.Structure N
    inst✝² : Countable (Sigma fun l => L.Functions l)
    inst✝¹ : Countable M
    inst✝ : Countable N
    hM : FirstOrder.Language.IsFraisseLimit K M
    hN : FirstOrder.Language.IsFraisseLimit K N
    ⊢ Nonempty (L.Equiv M N)
  -/
  let S : L.Substructure M := ⊥
  /-
    L : FirstOrder.Language
    K : Set (CategoryTheory.Bundled L.Structure)
    M : Type w
    inst✝⁴ : L.Structure M
    N : Type w
    inst✝³ : L.Structure N
    inst✝² : Countable (Sigma fun l => L.Functions l)
    inst✝¹ : Countable M
    inst✝ : Countable N
    hM : FirstOrder.Language.IsFraisseLimit K M
    hN : FirstOrder.Language.IsFraisseLimit K N
    S : L.Substructure M := Bot.bot
    ⊢ Nonempty (L.Equiv M N)
  -/
  have S_fg : FG L S := (fg_iff_structure_fg _).1 Substructure.fg_bot
  obtain ⟨_, ⟨emb_S : S ↪[L] N⟩⟩ : ⟨S, inferInstance⟩ ∈ L.age N := by
    rw [hN.age, ← hM.age]
    exact ⟨S_fg, ⟨subtype _⟩⟩
  let v : M ≃ₚ[L] N := {
    dom := S
    cod := emb_S.toHom.range
    toEquiv := emb_S.equivRange
  }
  exact ⟨Exists.choose (equiv_between_cg cg_of_countable cg_of_countable
    ⟨v, ((Substructure.fg_iff_structure_fg _).2 S_fg)⟩ (hM.isExtensionPair hN)
      (hN.isExtensionPair hM))⟩


/-- Any countable infinite structure in the empty language is a Fraïssé limit of the class of finite
structures. -/
theorem isFraisseLimit_of_countable_infinite
    (M : Type*) [Countable M] [Infinite M] [Language.empty.Structure M] :
    IsFraisseLimit { S : Bundled Language.empty.Structure | Finite S } M where
  age := by
    /-
      M : Type u_1
      inst✝² : Countable M
      inst✝¹ : Infinite M
      inst✝ : FirstOrder.Language.empty.Structure M
      ⊢ Eq (FirstOrder.Language.empty.age M) (setOf fun S => Finite ↑S)
    -/
    ext S
    /-
      case h
      M : Type u_1
      inst✝² : Countable M
      inst✝¹ : Infinite M
      inst✝ : FirstOrder.Language.empty.Structure M
      S : CategoryTheory.Bundled FirstOrder.Language.empty.Structure
      ⊢ Iff (Membership.mem (FirstOrder.Language.empty.age M) S) (Membership.mem (se …
    -/
    simp only [age, Structure.fg_iff_finite, mem_setOf_eq, and_iff_left_iff_imp]
    /-
      case h
      M : Type u_1
      inst✝² : Countable M
      inst✝¹ : Infinite M
      inst✝ : FirstOrder.Language.empty.Structure M
      S : CategoryTheory.Bundled FirstOrder.Language.empty.Structure
      ⊢ Finite ↑S → Nonempty (FirstOrder.Language.empty.Embedding (↑S) M)
    -/
    intro hS
    /-
      case h
      M : Type u_1
      inst✝² : Countable M
      inst✝¹ : Infinite M
      inst✝ : FirstOrder.Language.empty.Structure M
      S : CategoryTheory.Bundled FirstOrder.Language.empty.Structure
      hS : Finite ↑S
      ⊢ Nonempty (FirstOrder.Language.empty.Embedding (↑S) M)
    -/
    simp
    /-
      🎉 no goals
    -/
  ultrahomogeneous S hS f := by
    classical
    have : Finite S := hS.finite
    have : Infinite { x // x ∉ S } := ((Set.toFinite _).infinite_compl).to_subtype
    have : Finite f.toHom.range := (((Substructure.fg_iff_structure_fg S).1 hS).range _).finite
    have : Infinite { x // x ∉ f.toHom.range } := ((Set.toFinite _).infinite_compl ).to_subtype
    refine ⟨StrongHomClass.toEquiv (f.equivRange.subtypeCongr nonempty_equiv_of_countable.some), ?_⟩
    ext x
    simp [Equiv.subtypeCongr]


/-- The class of finite structures in the empty language is Fraïssé. -/
theorem isFraisse_finite : IsFraisse { S : Bundled.{w} Language.empty.Structure | Finite S } := by
  /-
    ⊢ FirstOrder.Language.IsFraisse (setOf fun S => Finite ↑S)
  -/
  have : Language.empty.Structure (ULift ℕ : Type w) := emptyStructure
  /-
    this : FirstOrder.Language.empty.Structure (ULift.{w, 0} Nat)
    ⊢ FirstOrder.Language.IsFraisse (setOf fun S => Finite ↑S)
  -/
  exact (isFraisseLimit_of_countable_infinite (ULift ℕ)).isFraisse
  /-
    🎉 no goals
  -/


