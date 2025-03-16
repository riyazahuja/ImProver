/-- A denumerable type is (constructively) bijective with `ℕ`. Typeclass equivalent of `α ≃ ℕ`. -/
class Denumerable (α : Type*) extends Encodable α where
  /-- `decode` and `encode` are inverses. -/
  decode_inv : ∀ n, ∃ a ∈ decode n, encode a = n


theorem decode_isSome (α) [Denumerable α] (n : ℕ) : (decode (α := α) n).isSome :=
  Option.isSome_iff_exists.2 <| (decode_inv n).imp fun _ => And.left


/-- Returns the `n`-th element of `α` indexed by the decoding. -/
def ofNat (α) [Denumerable α] (n : ℕ) : α :=
  Option.get _ (decode_isSome α n)


@[simp]
theorem decode_eq_ofNat (α) [Denumerable α] (n : ℕ) : decode (α := α) n = some (ofNat α n) :=
  Option.eq_some_of_isSome _


@[simp]
theorem ofNat_of_decode {n b} (h : decode (α := α) n = some b) : ofNat (α := α) n = b :=
  Option.some.inj <| (decode_eq_ofNat _ _).symm.trans h


@[simp]
theorem encode_ofNat (n) : encode (ofNat α n) = n := by
  /-
    α : Type u_1
    inst✝ : Denumerable α
    n : Nat
    ⊢ Eq (Encodable.encode (Denumerable.ofNat α n)) n
  -/
  obtain ⟨a, h, e⟩ := decode_inv (α := α) n
  /-
    case intro.intro
    α : Type u_1
    inst✝ : Denumerable α
    n : Nat
    a : α
    h : Membership.mem (Encodable.decode n) a
    e : Eq (Encodable.encode a) n
    ⊢ Eq (Encodable.encode (Denumerable.ofNat α n)) n
  -/
  rwa [ofNat_of_decode h]
  /-
    🎉 no goals
  -/


@[simp]
theorem ofNat_encode (a) : ofNat α (encode a) = a :=
  ofNat_of_decode (encodek _)


/-- A denumerable type is equivalent to `ℕ`. -/
def eqv (α) [Denumerable α] : α ≃ ℕ :=
  ⟨encode, ofNat α, ofNat_encode, encode_ofNat⟩

-- See Note [lower instance priority]

instance (priority := 100) : Infinite α :=
  Infinite.of_surjective _ (eqv α).surjective


/-- A type equivalent to `ℕ` is denumerable. -/
def mk' {α} (e : α ≃ ℕ) : Denumerable α where
  encode := e
  decode := some ∘ e.symm
  encodek _ := congr_arg some (e.symm_apply_apply _)
  decode_inv _ := ⟨_, rfl, e.apply_symm_apply _⟩


/-- Denumerability is conserved by equivalences. This is transitivity of equivalence the denumerable
way. -/
def ofEquiv (α) {β} [Denumerable α] (e : β ≃ α) : Denumerable β :=
  { Encodable.ofEquiv _ e with
    decode_inv := fun n => by
      -- Porting note: replaced `simp`
      simp_rw [Option.mem_def, decode_ofEquiv e, encode_ofEquiv e, decode_eq_ofNat,
        Option.map_some', Option.some_inj, exists_eq_left', Equiv.apply_symm_apply,
        Denumerable.encode_ofNat] }


@[simp]
theorem ofEquiv_ofNat (α) {β} [Denumerable α] (e : β ≃ α) (n) :
    @ofNat β (ofEquiv _ e) n = e.symm (ofNat α n) := by
  -- Porting note: added `letI`
  /-
    α : Type u_3
    β : Type u_4
    inst✝ : Denumerable α
    e : Equiv β α
    n : Nat
    ⊢ Eq (Denumerable.ofNat β n) (e.symm (Denumerable.ofNat α n))
  -/
  letI := ofEquiv _ e
  /-
    α : Type u_3
    β : Type u_4
    inst✝ : Denumerable α
    e : Equiv β α
    n : Nat
    this : Denumerable β := Denumerable.ofEquiv α e
    ⊢ Eq (Denumerable.ofNat β n) (e.symm (Denumerable.ofNat α n))
  -/
  refine ofNat_of_decode ?_
  /-
    α : Type u_3
    β : Type u_4
    inst✝ : Denumerable α
    e : Equiv β α
    n : Nat
    this : Denumerable β := Denumerable.ofEquiv α e
    ⊢ Eq (Encodable.decode n) (Option.some (e.symm (Denumerable.ofNat α n)))
  -/
  rw [decode_ofEquiv e]
  /-
    α : Type u_3
    β : Type u_4
    inst✝ : Denumerable α
    e : Equiv β α
    n : Nat
    this : Denumerable β := Denumerable.ofEquiv α e
    ⊢ Eq (Option.map (⇑e.symm) (Encodable.decode n)) (Option.some (e.symm (Denumer …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- All denumerable types are equivalent. -/
def equiv₂ (α β) [Denumerable α] [Denumerable β] : α ≃ β :=
  (eqv α).trans (eqv β).symm


instance nat : Denumerable ℕ :=
  ⟨fun _ => ⟨_, rfl, rfl⟩⟩


@[simp]
theorem ofNat_nat (n) : ofNat ℕ n = n :=
  rfl


/-- If `α` is denumerable, then so is `Option α`. -/
instance option : Denumerable (Option α) :=
  ⟨fun n => by
    cases n with
    | zero =>
      refine ⟨none, ?_, encode_none⟩
      rw [decode_option_zero, Option.mem_def]
    | succ n =>
      refine ⟨some (ofNat α n), ?_, ?_⟩
      · rw [decode_option_succ, decode_eq_ofNat, Option.map_some', Option.mem_def]
      rw [encode_some, encode_ofNat]⟩


/-- If `α` and `β` are denumerable, then so is their sum. -/
instance sum : Denumerable (α ⊕ β) :=
  ⟨fun n => by
    /-
      α : Type u_1
      β : Type u_2
      inst✝¹ : Denumerable α
      inst✝ : Denumerable β
      n : Nat
      ⊢ Exists fun a => And (Membership.mem (Encodable.decode n) a) (Eq (Encodable.e …
    -/
    suffices ∃ a ∈ @decodeSum α β _ _ n, encodeSum a = bit (bodd n) (div2 n) by simpa [bit_decomp]
    simp only [decodeSum, boddDiv2_eq, decode_eq_ofNat, Option.some.injEq, Option.map_some',
      Option.mem_def, Sum.exists]
    /-
      α : Type u_1
      β : Type u_2
      inst✝¹ : Denumerable α
      inst✝ : Denumerable β
      n : Nat
      ⊢ Or (Exists fun a => And (Eq (Encodable.decodeSum.match_1 (fun x => Option (S …
    -/
                     /-
                       🎉 no goals
                     -/
    cases bodd n <;> simp [decodeSum, bit, encodeSum, Nat.two_mul]⟩
                     /-
                       🎉 no goals
                     -/


/-- A denumerable collection of denumerable types is denumerable. -/
instance sigma : Denumerable (Sigma γ) :=
               /-
                 α : Type u_1
                 β : Type u_2
                 inst✝² : Denumerable α
                 inst✝¹ : Denumerable β
                 γ : α → Type u_3
                 inst✝ : (a : α) → Denumerable (γ a)
                 n : Nat
                 ⊢ Exists fun a => And (Membership.mem (Encodable.decode n) a) (Eq (Encodable.e …
               -/
  ⟨fun n => by simp [decodeSigma]⟩
               /-
                 🎉 no goals
               -/


@[simp]
theorem sigma_ofNat_val (n : ℕ) :
    ofNat (Sigma γ) n = ⟨ofNat α (unpair n).1, ofNat (γ _) (unpair n).2⟩ :=
                        /-
                          α : Type u_1
                          inst✝¹ : Denumerable α
                          γ : α → Type u_3
                          inst✝ : (a : α) → Denumerable (γ a)
                          n : Nat
                          ⊢ Eq (Option.some (Denumerable.ofNat (Sigma γ) n)) (Option.some ⟨Denumerable.o …
                        -/
  Option.some.inj <| by rw [← decode_eq_ofNat, decode_sigma_val]; simp
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


/-- If `α` and `β` are denumerable, then so is their product. -/
instance prod : Denumerable (α × β) :=
  ofEquiv _ (Equiv.sigmaEquivProd α β).symm

-- Porting note: removed @[simp] - simp can prove it

theorem prod_ofNat_val (n : ℕ) :
                                                                         /-
                                                                           α : Type u_1
                                                                           β : Type u_2
                                                                           inst✝¹ : Denumerable α
                                                                           inst✝ : Denumerable β
                                                                           n : Nat
                                                                           ⊢ Eq (Denumerable.ofNat (Prod α β) n) { fst := Denumerable.ofNat α (Nat.unpair …
                                                                         -/
    ofNat (α × β) n = (ofNat α (unpair n).1, ofNat β (unpair n).2) := by simp
                                                                         /-
                                                                           🎉 no goals
                                                                         -/


@[simp]
                                                      /-
                                                        ⊢ Eq (Denumerable.ofNat (Prod Nat Nat)) Nat.unpair
                                                      -/
theorem prod_nat_ofNat : ofNat (ℕ × ℕ) = unpair := by funext; simp
                                                              /-
                                                                🎉 no goals
                                                              -/


instance int : Denumerable ℤ :=
  Denumerable.mk' Equiv.intEquivNat


instance pnat : Denumerable ℕ+ :=
  Denumerable.mk' Equiv.pnatEquivNat


/-- The lift of a denumerable type is denumerable. -/
instance ulift : Denumerable (ULift α) :=
  ofEquiv _ Equiv.ulift


/-- The lift of a denumerable type is denumerable. -/
instance plift : Denumerable (PLift α) :=
  ofEquiv _ Equiv.plift


/-- If `α` is denumerable, then `α × α` and `α` are equivalent. -/
def pair : α × α ≃ α :=
  equiv₂ _ _


theorem exists_succ (x : s) : ∃ n, (x : ℕ) + n + 1 ∈ s := by
  /-
    s : Set Nat
    inst✝ : Infinite ↑s
    x : ↑s
    ⊢ Exists fun n => Membership.mem s (HAdd.hAdd (HAdd.hAdd (↑x) n) 1)
  -/
  by_contra h
  have : ∀ (a : ℕ) (_ : a ∈ s), a < x + 1 := fun a ha =>
    lt_of_not_ge fun hax => h ⟨a - (x + 1), by rwa [add_right_comm, Nat.add_sub_cancel' hax]⟩
  classical
  exact Fintype.false
    ⟨(((Multiset.range (succ x)).filter (· ∈ s)).pmap
      (fun (y : ℕ) (hy : y ∈ s) => Subtype.mk y hy) (by simp [-Multiset.range_succ])).toFinset,
      by simpa [Subtype.ext_iff_val, Multiset.mem_filter, -Multiset.range_succ] ⟩


/-- Returns the next natural in a set, according to the usual ordering of `ℕ`. -/
def succ (x : s) : s :=
  have h : ∃ m, (x : ℕ) + m + 1 ∈ s := exists_succ x
  ⟨↑x + Nat.find h + 1, Nat.find_spec h⟩


theorem succ_le_of_lt {x y : s} (h : y < x) : succ y ≤ x :=
  have hx : ∃ m, (y : ℕ) + m + 1 ∈ s := exists_succ _
  let ⟨k, hk⟩ := Nat.exists_eq_add_of_lt h
  have : Nat.find hx ≤ k := Nat.find_min' _ (hk ▸ x.2)
                                        /-
                                          s : Set Nat
                                          inst✝¹ : Infinite ↑s
                                          inst✝ : DecidablePred fun x => Membership.mem s x
                                          x y : ↑s
                                          h : LT.lt y x
                                          hx : Exists fun m => Membership.mem s (HAdd.hAdd (HAdd.hAdd (↑y) m) 1)
                                          k : Nat
                                          hk : Eq (↑x) (HAdd.hAdd (HAdd.hAdd (↑y) k) 1)
                                          this : LE.le (Nat.find hx) k
                                          ⊢ LE.le (HAdd.hAdd (HAdd.hAdd (↑y) (Nat.find hx)) 1) ↑x
                                        -/
  show (y : ℕ) + Nat.find hx + 1 ≤ x by omega
                                        /-
                                          🎉 no goals
                                        -/


theorem le_succ_of_forall_lt_le {x y : s} (h : ∀ z < x, z ≤ y) : x ≤ succ y :=
  have hx : ∃ m, (y : ℕ) + m + 1 ∈ s := exists_succ _
  show (x : ℕ) ≤ (y : ℕ) + Nat.find hx + 1 from
    le_of_not_gt fun hxy =>
      (h ⟨_, Nat.find_spec hx⟩ hxy).not_lt <|
            /-
              s : Set Nat
              inst✝¹ : Infinite ↑s
              inst✝ : DecidablePred fun x => Membership.mem s x
              x y : ↑s
              h : ∀ (z : ↑s), LT.lt z x → LE.le z y
              hx : Exists fun m => Membership.mem s (HAdd.hAdd (HAdd.hAdd (↑y) m) 1)
              hxy : GT.gt (↑x) (HAdd.hAdd (HAdd.hAdd (↑y) (Nat.find hx)) 1)
              ⊢ LT.lt (↑y) (HAdd.hAdd (HAdd.hAdd (↑y) (Nat.find hx)) 1)
            -/
        (by omega : (y : ℕ) < (y : ℕ) + Nat.find hx + 1)
            /-
              🎉 no goals
            -/


theorem lt_succ_self (x : s) : x < succ x :=
  calc
    -- Porting note: replaced `x + _`, added type annotations
    (x : ℕ) ≤ (x + Nat.find (exists_succ x) : ℕ) := le_add_right ..
    _ < (succ x : ℕ) := Nat.lt_succ_self (x + _)


theorem lt_succ_iff_le {x y : s} : x < succ y ↔ x ≤ y :=
  ⟨fun h => le_of_not_gt fun h' => not_le_of_gt h (succ_le_of_lt h'), fun h =>
    lt_of_le_of_lt h (lt_succ_self _)⟩


/-- Returns the `n`-th element of a set, according to the usual ordering of `ℕ`. -/
def ofNat (s : Set ℕ) [DecidablePred (· ∈ s)] [Infinite s] : ℕ → s
  | 0 => ⊥
  | n + 1 => succ (ofNat s n)


theorem ofNat_surjective : Surjective (ofNat s)
  | ⟨x, hx⟩ => by
    set t : List s :=
      ((List.range x).filter fun y => y ∈ s).pmap
        (fun (y : ℕ) (hy : y ∈ s) => ⟨y, hy⟩)
        (by intros a ha; simpa using (List.mem_filter.mp ha).2) with ht
    have hmt : ∀ {y : s}, y ∈ t ↔ y < ⟨x, hx⟩ := by
      simp [List.mem_filter, Subtype.ext_iff_val, ht]
    /-
      s : Set Nat
      inst✝¹ : Infinite ↑s
      inst✝ : DecidablePred fun x => Membership.mem s x
      x : Nat
      hx : Membership.mem s x
      t : List ↑s := List.pmap (fun y hy => ⟨y, hy⟩) (List.filter (fun y => Decidabl …
      ht : Eq t (List.pmap (fun y hy => ⟨y, hy⟩) (List.filter (fun y => Decidable.de …
      hmt : ∀ {y : ↑s}, Iff (Membership.mem t y) (LT.lt y ⟨x, hx⟩)
      ⊢ Exists fun a => Eq (Nat.Subtype.ofNat s a) ⟨x, hx⟩
    -/
    cases' hmax : List.maximum t with m
      /-
        case bot
        s : Set Nat
        inst✝¹ : Infinite ↑s
        inst✝ : DecidablePred fun x => Membership.mem s x
        x : Nat
        hx : Membership.mem s x
        t : List ↑s := List.pmap (fun y hy => ⟨y, hy⟩) (List.filter (fun y => Decidabl …
        ht : Eq t (List.pmap (fun y hy => ⟨y, hy⟩) (List.filter (fun y => Decidable.de …
        hmt : ∀ {y : ↑s}, Iff (Membership.mem t y) (LT.lt y ⟨x, hx⟩)
        hmax : Eq t.maximum Bot.bot
        ⊢ Exists fun a => Eq (Nat.Subtype.ofNat s a) ⟨x, hx⟩
      -/
    · refine ⟨0, le_antisymm bot_le (le_of_not_gt fun h => List.not_mem_nil (⊥ : s) ?_)⟩
      /-
        case bot
        s : Set Nat
        inst✝¹ : Infinite ↑s
        inst✝ : DecidablePred fun x => Membership.mem s x
        x : Nat
        hx : Membership.mem s x
        t : List ↑s := List.pmap (fun y hy => ⟨y, hy⟩) (List.filter (fun y => Decidabl …
        ht : Eq t (List.pmap (fun y hy => ⟨y, hy⟩) (List.filter (fun y => Decidable.de …
        hmt : ∀ {y : ↑s}, Iff (Membership.mem t y) (LT.lt y ⟨x, hx⟩)
        hmax : Eq t.maximum Bot.bot
        h : GT.gt ⟨x, hx⟩ (Nat.Subtype.ofNat s 0)
        ⊢ Membership.mem List.nil Bot.bot
      -/
      rwa [← List.maximum_eq_bot.1 hmax, hmt]
      /-
        🎉 no goals
      -/
    /-
      case coe
      s : Set Nat
      inst✝¹ : Infinite ↑s
      inst✝ : DecidablePred fun x => Membership.mem s x
      x : Nat
      hx : Membership.mem s x
      t : List ↑s := List.pmap (fun y hy => ⟨y, hy⟩) (List.filter (fun y => Decidabl …
      ht : Eq t (List.pmap (fun y hy => ⟨y, hy⟩) (List.filter (fun y => Decidable.de …
      hmt : ∀ {y : ↑s}, Iff (Membership.mem t y) (LT.lt y ⟨x, hx⟩)
      m : ↑s
      hmax : Eq t.maximum ↑m
      ⊢ Exists fun a => Eq (Nat.Subtype.ofNat s a) ⟨x, hx⟩
    -/
    have wf : ↑m < x := by simpa using hmt.mp (List.maximum_mem hmax)
    /-
      case coe
      s : Set Nat
      inst✝¹ : Infinite ↑s
      inst✝ : DecidablePred fun x => Membership.mem s x
      x : Nat
      hx : Membership.mem s x
      t : List ↑s := List.pmap (fun y hy => ⟨y, hy⟩) (List.filter (fun y => Decidabl …
      ht : Eq t (List.pmap (fun y hy => ⟨y, hy⟩) (List.filter (fun y => Decidable.de …
      hmt : ∀ {y : ↑s}, Iff (Membership.mem t y) (LT.lt y ⟨x, hx⟩)
      m : ↑s
      hmax : Eq t.maximum ↑m
      wf : LT.lt (↑m) x
      ⊢ Exists fun a => Eq (Nat.Subtype.ofNat s a) ⟨x, hx⟩
    -/
    rcases ofNat_surjective m with ⟨a, rfl⟩
    /-
      case coe.intro
      s : Set Nat
      inst✝¹ : Infinite ↑s
      inst✝ : DecidablePred fun x => Membership.mem s x
      x : Nat
      hx : Membership.mem s x
      t : List ↑s := List.pmap (fun y hy => ⟨y, hy⟩) (List.filter (fun y => Decidabl …
      ht : Eq t (List.pmap (fun y hy => ⟨y, hy⟩) (List.filter (fun y => Decidable.de …
      hmt : ∀ {y : ↑s}, Iff (Membership.mem t y) (LT.lt y ⟨x, hx⟩)
      a : Nat
      hmax : Eq t.maximum ↑(Nat.Subtype.ofNat s a)
      wf : LT.lt (↑(Nat.Subtype.ofNat s a)) x
      ⊢ Exists fun a => Eq (Nat.Subtype.ofNat s a) ⟨x, hx⟩
    -/
    refine ⟨a + 1, le_antisymm (succ_le_of_lt wf) ?_⟩
    /-
      case coe.intro
      s : Set Nat
      inst✝¹ : Infinite ↑s
      inst✝ : DecidablePred fun x => Membership.mem s x
      x : Nat
      hx : Membership.mem s x
      t : List ↑s := List.pmap (fun y hy => ⟨y, hy⟩) (List.filter (fun y => Decidabl …
      ht : Eq t (List.pmap (fun y hy => ⟨y, hy⟩) (List.filter (fun y => Decidable.de …
      hmt : ∀ {y : ↑s}, Iff (Membership.mem t y) (LT.lt y ⟨x, hx⟩)
      a : Nat
      hmax : Eq t.maximum ↑(Nat.Subtype.ofNat s a)
      wf : LT.lt (↑(Nat.Subtype.ofNat s a)) x
      ⊢ LE.le ⟨x, hx⟩ (Nat.Subtype.ofNat s (HAdd.hAdd a 1))
    -/
    exact le_succ_of_forall_lt_le fun z hz => List.le_maximum_of_mem (hmt.2 hz) hmax
    /-
      🎉 no goals
    -/
  termination_by n => n.val


@[simp]
theorem ofNat_range : Set.range (ofNat s) = Set.univ :=
  ofNat_surjective.range_eq


@[simp]
theorem coe_comp_ofNat_range : Set.range ((↑) ∘ ofNat s : ℕ → ℕ) = s := by
  /-
    s : Set Nat
    inst✝¹ : Infinite ↑s
    inst✝ : DecidablePred fun x => Membership.mem s x
    ⊢ Eq (Set.range (Function.comp Subtype.val (Nat.Subtype.ofNat s))) s
  -/
  rw [Set.range_comp Subtype.val, ofNat_range, Set.image_univ, Subtype.range_coe]
  /-
    🎉 no goals
  -/


private def toFunAux (x : s) : ℕ :=
  (List.range x).countP (· ∈ s)


private theorem toFunAux_eq {s : Set ℕ} [DecidablePred (· ∈ s)] (x : s) :
    toFunAux x = #{y ∈ Finset.range x | y ∈ s} := by
  /-
    s : Set Nat
    inst✝ : DecidablePred fun x => Membership.mem s x
    x : ↑s
    ⊢ Eq (Nat.Subtype.toFunAux x) (Finset.filter (fun y => Membership.mem s y) (Fi …
  -/
  rw [toFunAux, List.countP_eq_length_filter]
  /-
    s : Set Nat
    inst✝ : DecidablePred fun x => Membership.mem s x
    x : ↑s
    ⊢ Eq (List.filter (fun x => Decidable.decide (Membership.mem s x)) (List.range …
  -/
  rfl
  /-
    🎉 no goals
  -/


private theorem right_inverse_aux : ∀ n, toFunAux (ofNat s n) = n
  | 0 => by
    /-
      s : Set Nat
      inst✝¹ : Infinite ↑s
      inst✝ : DecidablePred fun x => Membership.mem s x
      ⊢ Eq (Nat.Subtype.toFunAux (Nat.Subtype.ofNat s 0)) 0
    -/
    rw [toFunAux_eq, card_eq_zero, eq_empty_iff_forall_not_mem]
    /-
      s : Set Nat
      inst✝¹ : Infinite ↑s
      inst✝ : DecidablePred fun x => Membership.mem s x
      ⊢ ∀ (x : Nat), Not (Membership.mem (Finset.filter (fun y => Membership.mem s y …
    -/
    rintro n hn
    /-
      s : Set Nat
      inst✝¹ : Infinite ↑s
      inst✝ : DecidablePred fun x => Membership.mem s x
      n : Nat
      hn : Membership.mem (Finset.filter (fun y => Membership.mem s y) (Finset.range …
      ⊢ False
    -/
    rw [mem_filter, ofNat, mem_range] at hn
    /-
      s : Set Nat
      inst✝¹ : Infinite ↑s
      inst✝ : DecidablePred fun x => Membership.mem s x
      n : Nat
      hn : And (LT.lt n ↑Bot.bot) (Membership.mem s n)
      ⊢ False
    -/
    exact bot_le.not_lt (show (⟨n, hn.2⟩ : s) < ⊥ from hn.1)
    /-
      🎉 no goals
    -/
  | n + 1 => by
    /-
      s : Set Nat
      inst✝¹ : Infinite ↑s
      inst✝ : DecidablePred fun x => Membership.mem s x
      n : Nat
      ⊢ Eq (Nat.Subtype.toFunAux (Nat.Subtype.ofNat s (HAdd.hAdd n 1))) (HAdd.hAdd n …
    -/
    have ih : toFunAux (ofNat s n) = n := right_inverse_aux n
    /-
      s : Set Nat
      inst✝¹ : Infinite ↑s
      inst✝ : DecidablePred fun x => Membership.mem s x
      n : Nat
      ih : Eq (Nat.Subtype.toFunAux (Nat.Subtype.ofNat s n)) n
      ⊢ Eq (Nat.Subtype.toFunAux (Nat.Subtype.ofNat s (HAdd.hAdd n 1))) (HAdd.hAdd n …
    -/
    have h₁ : (ofNat s n : ℕ) ∉ {x ∈ range (ofNat s n) | x ∈ s} := by simp
    have h₂ : {x ∈ range (succ (ofNat s n)) | x ∈ s} =
        insert ↑(ofNat s n) {x ∈ range (ofNat s n) | x ∈ s} := by
      simp only [Finset.ext_iff, mem_insert, mem_range, mem_filter]
      exact fun m =>
        ⟨fun h => by
          simp only [h.2, and_true]
          exact Or.symm (lt_or_eq_of_le ((@lt_succ_iff_le _ _ _ ⟨m, h.2⟩ _).1 h.1)),
         fun h =>
          h.elim (fun h => h.symm ▸ ⟨lt_succ_self _, (ofNat s n).prop⟩) fun h =>
            ⟨h.1.trans (lt_succ_self _), h.2⟩⟩
    /-
      s : Set Nat
      inst✝¹ : Infinite ↑s
      inst✝ : DecidablePred fun x => Membership.mem s x
      n : Nat
      ih : Eq (Nat.Subtype.toFunAux (Nat.Subtype.ofNat s n)) n
      h₁ : Not (Membership.mem (Finset.filter (fun x => Membership.mem s x) (Finset. …
      h₂ : Eq (Finset.filter (fun x => Membership.mem s x) (Finset.range ↑(Nat.Subty …
      ⊢ Eq (Nat.Subtype.toFunAux (Nat.Subtype.ofNat s (HAdd.hAdd n 1))) (HAdd.hAdd n …
    -/
    simp only [toFunAux_eq, ofNat, range_succ] at ih ⊢
    conv =>
      rhs
      rw [← ih, ← card_insert_of_not_mem h₁, ← h₂]


/-- Any infinite set of naturals is denumerable. -/
def denumerable (s : Set ℕ) [DecidablePred (· ∈ s)] [Infinite s] : Denumerable s :=
  Denumerable.ofEquiv ℕ
    { toFun := toFunAux
      invFun := ofNat s
      left_inv := leftInverse_of_surjective_of_rightInverse ofNat_surjective right_inverse_aux
      right_inv := right_inverse_aux }


/-- An infinite encodable type is denumerable. -/
def ofEncodableOfInfinite (α : Type*) [Encodable α] [Infinite α] : Denumerable α := by
  /-
    α✝ : Type u_1
    β : Type u_2
    α : Type u_3
    inst✝¹ : Encodable α
    inst✝ : Infinite α
    ⊢ Denumerable α
  -/
  letI := @decidableRangeEncode α _
  letI : Infinite (Set.range (@encode α _)) :=
    Infinite.of_injective _ (Equiv.ofInjective _ encode_injective).injective
  /-
    α✝ : Type u_1
    β : Type u_2
    α : Type u_3
    inst✝¹ : Encodable α
    inst✝ : Infinite α
    this✝ : DecidablePred fun x => Membership.mem (Set.range Encodable.encode) x : …
    this : Infinite ↑(Set.range Encodable.encode) := Infinite.of_injective (⇑(Equi …
    ⊢ Denumerable α
  -/
  letI := Nat.Subtype.denumerable (Set.range (@encode α _))
  /-
    α✝ : Type u_1
    β : Type u_2
    α : Type u_3
    inst✝¹ : Encodable α
    inst✝ : Infinite α
    this✝¹ : DecidablePred fun x => Membership.mem (Set.range Encodable.encode) x  …
    this✝ : Infinite ↑(Set.range Encodable.encode) := Infinite.of_injective (⇑(Equ …
    this : Denumerable ↑(Set.range Encodable.encode) := Nat.Subtype.denumerable (S …
    ⊢ Denumerable α
  -/
  exact Denumerable.ofEquiv (Set.range (@encode α _)) (equivRangeEncode α)
  /-
    🎉 no goals
  -/


/-- See also `nonempty_encodable`, `nonempty_fintype`. -/
theorem nonempty_denumerable (α : Type*) [Countable α] [Infinite α] : Nonempty (Denumerable α) :=
  (nonempty_encodable α).map fun h => @Denumerable.ofEncodableOfInfinite _ h _


theorem nonempty_denumerable_iff {α : Type*} :
    Nonempty (Denumerable α) ↔ Countable α ∧ Infinite α :=
  ⟨fun ⟨_⟩ ↦ ⟨inferInstance, inferInstance⟩, fun ⟨_, _⟩ ↦ nonempty_denumerable _⟩


instance nonempty_equiv_of_countable [Countable α] [Infinite α] [Countable β] [Infinite β] :
    Nonempty (α ≃ β) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : Countable α
    inst✝² : Infinite α
    inst✝¹ : Countable β
    inst✝ : Infinite β
    ⊢ Nonempty (Equiv α β)
  -/
  cases nonempty_denumerable α
  /-
    case intro
    α : Type u_1
    β : Type u_2
    inst✝³ : Countable α
    inst✝² : Infinite α
    inst✝¹ : Countable β
    inst✝ : Infinite β
    val✝ : Denumerable α
    ⊢ Nonempty (Equiv α β)
  -/
  cases nonempty_denumerable β
  /-
    case intro.intro
    α : Type u_1
    β : Type u_2
    inst✝³ : Countable α
    inst✝² : Infinite α
    inst✝¹ : Countable β
    inst✝ : Infinite β
    val✝¹ : Denumerable α
    val✝ : Denumerable β
    ⊢ Nonempty (Equiv α β)
  -/
  exact ⟨(Denumerable.eqv _).trans (Denumerable.eqv _).symm⟩
  /-
    🎉 no goals
  -/

