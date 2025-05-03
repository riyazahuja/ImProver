/-- A module of finite length is either trivial or a simple extension of a module known
to be of finite length. -/
inductive IsFiniteLength : ∀ (M : Type u) [AddCommGroup M] [Module R M], Prop
  | of_subsingleton {M} [AddCommGroup M] [Module R M] [Subsingleton M] : IsFiniteLength M
  | of_simple_quotient {M} [AddCommGroup M] [Module R M] {N : Submodule R M}
      [IsSimpleModule R (M ⧸ N)] : IsFiniteLength N → IsFiniteLength M


theorem LinearEquiv.isFiniteLength (e : M ≃ₗ[R] N)
    (h : IsFiniteLength R M) : IsFiniteLength R N := by
  /-
    R : Type u_1
    inst✝⁴ : Ring R
    M : Type u_2
    N : Type u_3
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    e : LinearEquiv (RingHom.id R) M N
    h : IsFiniteLength R M
    ⊢ IsFiniteLength R N
  -/
  induction' h with M _ _ _ M _ _ S _ _ ih generalizing N
    /-
      case of_subsingleton
      R : Type u_1
      inst✝⁷ : Ring R
      M✝ : Type u_2
      inst✝⁶ : AddCommGroup M✝
      inst✝⁵ : Module R M✝
      M : Type u_2
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      inst✝² : Subsingleton M
      N : Type u_3
      inst✝¹ : AddCommGroup N
      inst✝ : Module R N
      e : LinearEquiv (RingHom.id R) M N
      ⊢ IsFiniteLength R N
    -/
  · have := e.symm.toEquiv.subsingleton; exact .of_subsingleton
                                         /-
                                           🎉 no goals
                                         -/
  · have : IsSimpleModule R (N ⧸ Submodule.map (e : M →ₗ[R] N) S) :=
      IsSimpleModule.congr (Submodule.Quotient.equiv S _ e rfl).symm
    /-
      case of_simple_quotient
      R : Type u_1
      inst✝⁷ : Ring R
      M✝ : Type u_2
      inst✝⁶ : AddCommGroup M✝
      inst✝⁵ : Module R M✝
      M : Type u_2
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      S : Submodule R M
      inst✝² : IsSimpleModule R (HasQuotient.Quotient M S)
      a✝ : IsFiniteLength R (Subtype fun x => Membership.mem S x)
      ih : ∀ {N : Type u_3} [inst : AddCommGroup N] [inst_1 : Module R N], LinearEqu …
      N : Type u_3
      inst✝¹ : AddCommGroup N
      inst✝ : Module R N
      e : LinearEquiv (RingHom.id R) M N
      this : IsSimpleModule R (HasQuotient.Quotient N (Submodule.map (↑e) S))
      ⊢ IsFiniteLength R N
    -/
    exact .of_simple_quotient (ih <| e.submoduleMap S)
    /-
      🎉 no goals
    -/


variable (R M) in
theorem exists_compositionSeries_of_isNoetherian_isArtinian [IsNoetherian R M] [IsArtinian R M] :
    ∃ s : CompositionSeries (Submodule R M), s.head = ⊥ ∧ s.last = ⊤ := by
  /-
    R : Type u_1
    inst✝⁴ : Ring R
    M : Type u_2
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : IsNoetherian R M
    inst✝ : IsArtinian R M
    ⊢ Exists fun s => And (Eq (RelSeries.head s) Bot.bot) (Eq (RelSeries.last s) T …
  -/
  obtain ⟨f, f0, n, hn⟩ := exists_covBy_seq_of_wellFoundedLT_wellFoundedGT (Submodule R M)
  /-
    case intro.intro.intro
    R : Type u_1
    inst✝⁴ : Ring R
    M : Type u_2
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : IsNoetherian R M
    inst✝ : IsArtinian R M
    f : Nat → Submodule R M
    f0 : IsMin (f 0)
    n : Nat
    hn : And (IsMax (f n)) (∀ (i : Nat), LT.lt i n → CovBy (f i) (f (HAdd.hAdd i 1 …
    ⊢ Exists fun s => And (Eq (RelSeries.head s) Bot.bot) (Eq (RelSeries.last s) T …
  -/
  exact ⟨⟨n, fun i ↦ f i, fun i ↦ hn.2 i i.2⟩, f0.eq_bot, hn.1.eq_top⟩
  /-
    🎉 no goals
  -/


theorem isFiniteLength_of_exists_compositionSeries
    (h : ∃ s : CompositionSeries (Submodule R M), s.head = ⊥ ∧ s.last = ⊤) :
    IsFiniteLength R M :=
  Submodule.topEquiv.isFiniteLength <| by
    /-
      R : Type u_1
      inst✝² : Ring R
      M : Type u_2
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      h : Exists fun s => And (Eq (RelSeries.head s) Bot.bot) (Eq (RelSeries.last s) …
      ⊢ IsFiniteLength R (Subtype fun x => Membership.mem Top.top x)
    -/
    obtain ⟨s, s_head, s_last⟩ := h
    /-
      case intro.intro
      R : Type u_1
      inst✝² : Ring R
      M : Type u_2
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      s : CompositionSeries (Submodule R M)
      s_head : Eq (RelSeries.head s) Bot.bot
      s_last : Eq (RelSeries.last s) Top.top
      ⊢ IsFiniteLength R (Subtype fun x => Membership.mem Top.top x)
    -/
    rw [← s_last]
    /-
      case intro.intro
      R : Type u_1
      inst✝² : Ring R
      M : Type u_2
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      s : CompositionSeries (Submodule R M)
      s_head : Eq (RelSeries.head s) Bot.bot
      s_last : Eq (RelSeries.last s) Top.top
      ⊢ IsFiniteLength R (Subtype fun x => Membership.mem (RelSeries.last s) x)
    -/
    suffices ∀ i, IsFiniteLength R (s i) from this (Fin.last _)
    /-
      case intro.intro
      R : Type u_1
      inst✝² : Ring R
      M : Type u_2
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      s : CompositionSeries (Submodule R M)
      s_head : Eq (RelSeries.head s) Bot.bot
      s_last : Eq (RelSeries.last s) Top.top
      ⊢ ∀ (i : Fin (HAdd.hAdd s.length 1)), IsFiniteLength R (Subtype fun x => Membe …
    -/
    intro i
    /-
      case intro.intro
      R : Type u_1
      inst✝² : Ring R
      M : Type u_2
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      s : CompositionSeries (Submodule R M)
      s_head : Eq (RelSeries.head s) Bot.bot
      s_last : Eq (RelSeries.last s) Top.top
      i : Fin (HAdd.hAdd s.length 1)
      ⊢ IsFiniteLength R (Subtype fun x => Membership.mem (s.toFun i) x)
    -/
    induction' i using Fin.induction with i ih
      /-
        case intro.intro.zero
        R : Type u_1
        inst✝² : Ring R
        M : Type u_2
        inst✝¹ : AddCommGroup M
        inst✝ : Module R M
        s : CompositionSeries (Submodule R M)
        s_head : Eq (RelSeries.head s) Bot.bot
        s_last : Eq (RelSeries.last s) Top.top
        ⊢ IsFiniteLength R (Subtype fun x => Membership.mem (s.toFun 0) x)
      -/
    · change IsFiniteLength R s.head; rw [s_head]; exact .of_subsingleton
                                                   /-
                                                     🎉 no goals
                                                   -/
    /-
      case intro.intro.succ
      R : Type u_1
      inst✝² : Ring R
      M : Type u_2
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      s : CompositionSeries (Submodule R M)
      s_head : Eq (RelSeries.head s) Bot.bot
      s_last : Eq (RelSeries.last s) Top.top
      i : Fin s.length
      ih : IsFiniteLength R (Subtype fun x => Membership.mem (s.toFun i.castSucc) x)
      ⊢ IsFiniteLength R (Subtype fun x => Membership.mem (s.toFun i.succ) x)
    -/
    let cov := s.step i
    /-
      case intro.intro.succ
      R : Type u_1
      inst✝² : Ring R
      M : Type u_2
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      s : CompositionSeries (Submodule R M)
      s_head : Eq (RelSeries.head s) Bot.bot
      s_last : Eq (RelSeries.last s) Top.top
      i : Fin s.length
      ih : IsFiniteLength R (Subtype fun x => Membership.mem (s.toFun i.castSucc) x)
      cov : JordanHolderLattice.IsMaximal (s.toFun i.castSucc) (s.toFun i.succ) := s …
      ⊢ IsFiniteLength R (Subtype fun x => Membership.mem (s.toFun i.succ) x)
    -/
    have := (covBy_iff_quot_is_simple cov.le).mp cov
    have := ((s i.castSucc).comap (s i.succ).subtype).equivMapOfInjective
      _ (Submodule.injective_subtype _)
    /-
      case intro.intro.succ
      R : Type u_1
      inst✝² : Ring R
      M : Type u_2
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      s : CompositionSeries (Submodule R M)
      s_head : Eq (RelSeries.head s) Bot.bot
      s_last : Eq (RelSeries.last s) Top.top
      i : Fin s.length
      ih : IsFiniteLength R (Subtype fun x => Membership.mem (s.toFun i.castSucc) x)
      cov : JordanHolderLattice.IsMaximal (s.toFun i.castSucc) (s.toFun i.succ) := s …
      this✝ : IsSimpleModule R (HasQuotient.Quotient (Subtype fun x => Membership.me …
      this : LinearEquiv (RingHom.id R) (Subtype fun x => Membership.mem (Submodule. …
      ⊢ IsFiniteLength R (Subtype fun x => Membership.mem (s.toFun i.succ) x)
    -/
    rw [Submodule.map_comap_subtype, inf_of_le_right cov.le] at this
    /-
      case intro.intro.succ
      R : Type u_1
      inst✝² : Ring R
      M : Type u_2
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      s : CompositionSeries (Submodule R M)
      s_head : Eq (RelSeries.head s) Bot.bot
      s_last : Eq (RelSeries.last s) Top.top
      i : Fin s.length
      ih : IsFiniteLength R (Subtype fun x => Membership.mem (s.toFun i.castSucc) x)
      cov : JordanHolderLattice.IsMaximal (s.toFun i.castSucc) (s.toFun i.succ) := s …
      this✝ : IsSimpleModule R (HasQuotient.Quotient (Subtype fun x => Membership.me …
      this : LinearEquiv (RingHom.id R) (Subtype fun x => Membership.mem (Submodule. …
      ⊢ IsFiniteLength R (Subtype fun x => Membership.mem (s.toFun i.succ) x)
    -/
    exact .of_simple_quotient (this.symm.isFiniteLength ih)
    /-
      🎉 no goals
    -/


theorem isFiniteLength_iff_isNoetherian_isArtinian :
    IsFiniteLength R M ↔ IsNoetherian R M ∧ IsArtinian R M :=
  ⟨fun h ↦ h.rec (fun {M} _ _ _ ↦ ⟨inferInstance, inferInstance⟩) fun M _ _ {N} _ _ ⟨_, _⟩ ↦
    ⟨(isNoetherian_iff_submodule_quotient N).mpr ⟨‹_›, isNoetherian_iff'.mpr inferInstance⟩,
      (isArtinian_iff_submodule_quotient N).mpr ⟨‹_›, inferInstance⟩⟩,
    fun ⟨_, _⟩ ↦ isFiniteLength_of_exists_compositionSeries
      (exists_compositionSeries_of_isNoetherian_isArtinian R M)⟩


theorem isFiniteLength_iff_exists_compositionSeries :
    IsFiniteLength R M ↔ ∃ s : CompositionSeries (Submodule R M), s.head = ⊥ ∧ s.last = ⊤ :=
  ⟨fun h ↦ have ⟨_, _⟩ := isFiniteLength_iff_isNoetherian_isArtinian.mp h
    exists_compositionSeries_of_isNoetherian_isArtinian R M,
    isFiniteLength_of_exists_compositionSeries⟩


theorem IsSemisimpleModule.finite_tfae [IsSemisimpleModule R M] :
    List.TFAE [Module.Finite R M, IsNoetherian R M, IsArtinian R M, IsFiniteLength R M,
      ∃ s : Set (Submodule R M), s.Finite ∧ sSupIndep s ∧
        sSup s = ⊤ ∧ ∀ m ∈ s, IsSimpleModule R m] := by
  /-
    R : Type u_1
    inst✝³ : Ring R
    M : Type u_2
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : IsSemisimpleModule R M
    ⊢ (List.cons (Module.Finite R M) (List.cons (IsNoetherian R M) (List.cons (IsA …
  -/
  rw [isFiniteLength_iff_isNoetherian_isArtinian]
  /-
    R : Type u_1
    inst✝³ : Ring R
    M : Type u_2
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : IsSemisimpleModule R M
    ⊢ (List.cons (Module.Finite R M) (List.cons (IsNoetherian R M) (List.cons (IsA …
  -/
  obtain ⟨s, hs⟩ := IsSemisimpleModule.exists_sSupIndep_sSup_simples_eq_top R M
  /-
    case intro
    R : Type u_1
    inst✝³ : Ring R
    M : Type u_2
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : IsSemisimpleModule R M
    s : Set (Submodule R M)
    hs : And (sSupIndep s) (And (Eq (SupSet.sSup s) Top.top) (∀ (m : Submodule R M …
    ⊢ (List.cons (Module.Finite R M) (List.cons (IsNoetherian R M) (List.cons (IsA …
  -/
  tfae_have 1 ↔ 2 := ⟨fun _ ↦ inferInstance, fun _ ↦ inferInstance⟩
  /-
    case intro
    R : Type u_1
    inst✝³ : Ring R
    M : Type u_2
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : IsSemisimpleModule R M
    s : Set (Submodule R M)
    hs : And (sSupIndep s) (And (Eq (SupSet.sSup s) Top.top) (∀ (m : Submodule R M …
    tfae_1_iff_2 : Iff (Module.Finite R M) (IsNoetherian R M)
    ⊢ (List.cons (Module.Finite R M) (List.cons (IsNoetherian R M) (List.cons (IsA …
  -/
  tfae_have 2 → 5 := fun _ ↦ ⟨s, WellFoundedGT.finite_of_sSupIndep hs.1, hs⟩
  /-
    case intro
    R : Type u_1
    inst✝³ : Ring R
    M : Type u_2
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : IsSemisimpleModule R M
    s : Set (Submodule R M)
    hs : And (sSupIndep s) (And (Eq (SupSet.sSup s) Top.top) (∀ (m : Submodule R M …
    tfae_1_iff_2 : Iff (Module.Finite R M) (IsNoetherian R M)
    tfae_2_to_5 : IsNoetherian R M → Exists fun s => And s.Finite (And (sSupIndep  …
    ⊢ (List.cons (Module.Finite R M) (List.cons (IsNoetherian R M) (List.cons (IsA …
  -/
  tfae_have 3 → 5 := fun _ ↦ ⟨s, WellFoundedLT.finite_of_sSupIndep hs.1, hs⟩
  tfae_have 5 → 4 := fun ⟨s, fin, _, sSup_eq_top, simple⟩ ↦ by
    rw [← isNoetherian_top_iff, ← Submodule.topEquiv.isArtinian_iff,
      ← sSup_eq_top, sSup_eq_iSup, ← iSup_subtype'']
    rw [SetCoe.forall'] at simple
    have := fin.to_subtype
    exact ⟨isNoetherian_iSup, isArtinian_iSup⟩
  /-
    case intro
    R : Type u_1
    inst✝³ : Ring R
    M : Type u_2
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : IsSemisimpleModule R M
    s : Set (Submodule R M)
    hs : And (sSupIndep s) (And (Eq (SupSet.sSup s) Top.top) (∀ (m : Submodule R M …
    tfae_1_iff_2 : Iff (Module.Finite R M) (IsNoetherian R M)
    tfae_2_to_5 : IsNoetherian R M → Exists fun s => And s.Finite (And (sSupIndep  …
    tfae_3_to_5 : IsArtinian R M → Exists fun s => And s.Finite (And (sSupIndep s) …
    tfae_5_to_4 : (Exists fun s => And s.Finite (And (sSupIndep s) (And (Eq (SupSe …
    ⊢ (List.cons (Module.Finite R M) (List.cons (IsNoetherian R M) (List.cons (IsA …
  -/
  tfae_have 4 → 2 := And.left
  /-
    case intro
    R : Type u_1
    inst✝³ : Ring R
    M : Type u_2
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : IsSemisimpleModule R M
    s : Set (Submodule R M)
    hs : And (sSupIndep s) (And (Eq (SupSet.sSup s) Top.top) (∀ (m : Submodule R M …
    tfae_1_iff_2 : Iff (Module.Finite R M) (IsNoetherian R M)
    tfae_2_to_5 : IsNoetherian R M → Exists fun s => And s.Finite (And (sSupIndep  …
    tfae_3_to_5 : IsArtinian R M → Exists fun s => And s.Finite (And (sSupIndep s) …
    tfae_5_to_4 : (Exists fun s => And s.Finite (And (sSupIndep s) (And (Eq (SupSe …
    tfae_4_to_2 : And (IsNoetherian R M) (IsArtinian R M) → IsNoetherian R M
    ⊢ (List.cons (Module.Finite R M) (List.cons (IsNoetherian R M) (List.cons (IsA …
  -/
  tfae_have 4 → 3 := And.right
  /-
    case intro
    R : Type u_1
    inst✝³ : Ring R
    M : Type u_2
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : IsSemisimpleModule R M
    s : Set (Submodule R M)
    hs : And (sSupIndep s) (And (Eq (SupSet.sSup s) Top.top) (∀ (m : Submodule R M …
    tfae_1_iff_2 : Iff (Module.Finite R M) (IsNoetherian R M)
    tfae_2_to_5 : IsNoetherian R M → Exists fun s => And s.Finite (And (sSupIndep  …
    tfae_3_to_5 : IsArtinian R M → Exists fun s => And s.Finite (And (sSupIndep s) …
    tfae_5_to_4 : (Exists fun s => And s.Finite (And (sSupIndep s) (And (Eq (SupSe …
    tfae_4_to_2 : And (IsNoetherian R M) (IsArtinian R M) → IsNoetherian R M
    tfae_4_to_3 : And (IsNoetherian R M) (IsArtinian R M) → IsArtinian R M
    ⊢ (List.cons (Module.Finite R M) (List.cons (IsNoetherian R M) (List.cons (IsA …
  -/
  tfae_finish
  /-
    🎉 no goals
  -/


instance [IsSemisimpleModule R M] [Module.Finite R M] : IsArtinian R M :=
   /-
     R : Type u_1
     inst✝⁶ : Ring R
     M : Type u_2
     N : Type u_3
     inst✝⁵ : AddCommGroup M
     inst✝⁴ : Module R M
     inst✝³ : AddCommGroup N
     inst✝² : Module R N
     inst✝¹ : IsSemisimpleModule R M
     inst✝ : Module.Finite R M
     ⊢ Eq ((List.cons (Module.Finite ?m.72255 ?m.72286) (List.cons (IsNoetherian ?m …
   -/
   /-
     🎉 no goals
   -/
  (IsSemisimpleModule.finite_tfae.out 0 2).mp ‹_›
   /-
     🎉 no goals
   -/

/- The following instances are now automatic:
example [IsSemisimpleRing R] : IsNoetherianRing R := inferInstance
example [IsSemisimpleRing R] : IsArtinianRing R := inferInstance
-/

