private def lbp (m n : ℕ) : Prop :=
  m = n + 1 ∧ ∀ k ≤ n, false ∈ p k


private def wf_lbp (H : ∃ n, true ∈ p n ∧ ∀ k < n, (p k).Dom) : WellFounded (lbp p) :=
  ⟨by
    /-
      p : PFun Nat Bool
      H : Exists fun n => And (Membership.mem (p n) Bool.true) (∀ (k : Nat), LT.lt k …
      ⊢ ∀ (a : Nat), Acc (Nat.lbp p) a
    -/
    let ⟨n, pn⟩ := H
    /-
      p : PFun Nat Bool
      H : Exists fun n => And (Membership.mem (p n) Bool.true) (∀ (k : Nat), LT.lt k …
      n : Nat
      pn : And (Membership.mem (p n) Bool.true) (∀ (k : Nat), LT.lt k n → (p k).Dom)
      ⊢ ∀ (a : Nat), Acc (Nat.lbp p) a
    -/
    suffices ∀ m k, n ≤ k + m → Acc (lbp p) k by exact fun a => this _ _ (Nat.le_add_left _ _)
    /-
      p : PFun Nat Bool
      H : Exists fun n => And (Membership.mem (p n) Bool.true) (∀ (k : Nat), LT.lt k …
      n : Nat
      pn : And (Membership.mem (p n) Bool.true) (∀ (k : Nat), LT.lt k n → (p k).Dom)
      ⊢ ∀ (m k : Nat), LE.le n (HAdd.hAdd k m) → Acc (Nat.lbp p) k
    -/
    intro m k kn
    /-
      p : PFun Nat Bool
      H : Exists fun n => And (Membership.mem (p n) Bool.true) (∀ (k : Nat), LT.lt k …
      n : Nat
      pn : And (Membership.mem (p n) Bool.true) (∀ (k : Nat), LT.lt k n → (p k).Dom)
      m k : Nat
      kn : LE.le n (HAdd.hAdd k m)
      ⊢ Acc (Nat.lbp p) k
    -/
    induction' m with m IH generalizing k <;> refine ⟨_, fun y r => ?_⟩ <;> rcases r with ⟨rfl, a⟩
      /-
        case zero.intro
        p : PFun Nat Bool
        H : Exists fun n => And (Membership.mem (p n) Bool.true) (∀ (k : Nat), LT.lt k …
        n : Nat
        pn : And (Membership.mem (p n) Bool.true) (∀ (k : Nat), LT.lt k n → (p k).Dom)
        k : Nat
        kn : LE.le n (HAdd.hAdd k 0)
        a : ∀ (k_1 : Nat), LE.le k_1 k → Membership.mem (p k_1) Bool.false
        ⊢ Acc (Nat.lbp p) (HAdd.hAdd k 1)
      -/
    · injection mem_unique pn.1 (a _ kn)
      /-
        🎉 no goals
      -/
      /-
        case succ.intro
        p : PFun Nat Bool
        H : Exists fun n => And (Membership.mem (p n) Bool.true) (∀ (k : Nat), LT.lt k …
        n : Nat
        pn : And (Membership.mem (p n) Bool.true) (∀ (k : Nat), LT.lt k n → (p k).Dom)
        m : Nat
        IH : ∀ (k : Nat), LE.le n (HAdd.hAdd k m) → Acc (Nat.lbp p) k
        k : Nat
        kn : LE.le n (HAdd.hAdd k (HAdd.hAdd m 1))
        a : ∀ (k_1 : Nat), LE.le k_1 k → Membership.mem (p k_1) Bool.false
        ⊢ Acc (Nat.lbp p) (HAdd.hAdd k 1)
      -/
    · exact IH _ (by rw [Nat.add_right_comm]; exact kn)⟩
      /-
        🎉 no goals
      -/


def rfindX : { n // true ∈ p n ∧ ∀ m < n, false ∈ p m } :=
  suffices ∀ k, (∀ n < k, false ∈ p n) → { n // true ∈ p n ∧ ∀ m < n, false ∈ p m } from
    this 0 fun _ => (Nat.not_lt_zero _).elim
  @WellFounded.fix _ _ (lbp p) (wf_lbp p H)
    (by
      /-
        p : PFun Nat Bool
        H : Exists fun n => And (Membership.mem (p n) Bool.true) (∀ (k : Nat), LT.lt k …
        ⊢ (x : Nat) → ((y : Nat) → Nat.lbp p y x → (∀ (n : Nat), LT.lt n y → Membershi …
      -/
      intro m IH al
      have pm : (p m).Dom := by
        rcases H with ⟨n, h₁, h₂⟩
        rcases lt_trichotomy m n with (h₃ | h₃ | h₃)
        · exact h₂ _ h₃
        · rw [h₃]
          exact h₁.fst
        · injection mem_unique h₁ (al _ h₃)
      /-
        p : PFun Nat Bool
        H : Exists fun n => And (Membership.mem (p n) Bool.true) (∀ (k : Nat), LT.lt k …
        m : Nat
        IH : (y : Nat) → Nat.lbp p y m → (∀ (n : Nat), LT.lt n y → Membership.mem (p n …
        al : ∀ (n : Nat), LT.lt n m → Membership.mem (p n) Bool.false
        pm : (p m).Dom
        ⊢ Subtype fun n => And (Membership.mem (p n) Bool.true) (∀ (m : Nat), LT.lt m  …
      -/
      cases e : (p m).get pm
        /-
          case false
          p : PFun Nat Bool
          H : Exists fun n => And (Membership.mem (p n) Bool.true) (∀ (k : Nat), LT.lt k …
          m : Nat
          IH : (y : Nat) → Nat.lbp p y m → (∀ (n : Nat), LT.lt n y → Membership.mem (p n …
          al : ∀ (n : Nat), LT.lt n m → Membership.mem (p n) Bool.false
          pm : (p m).Dom
          e : Eq ((p m).get pm) Bool.false
          ⊢ Subtype fun n => And (Membership.mem (p n) Bool.true) (∀ (m : Nat), LT.lt m  …
        -/
      · suffices ∀ᵉ k ≤ m, false ∈ p k from IH _ ⟨rfl, this⟩ fun n h => this _ (le_of_lt_succ h)
        /-
          case false
          p : PFun Nat Bool
          H : Exists fun n => And (Membership.mem (p n) Bool.true) (∀ (k : Nat), LT.lt k …
          m : Nat
          IH : (y : Nat) → Nat.lbp p y m → (∀ (n : Nat), LT.lt n y → Membership.mem (p n …
          al : ∀ (n : Nat), LT.lt n m → Membership.mem (p n) Bool.false
          pm : (p m).Dom
          e : Eq ((p m).get pm) Bool.false
          ⊢ ∀ (k : Nat), LE.le k m → Membership.mem (p k) Bool.false
        -/
        intro n h
        /-
          case false
          p : PFun Nat Bool
          H : Exists fun n => And (Membership.mem (p n) Bool.true) (∀ (k : Nat), LT.lt k …
          m : Nat
          IH : (y : Nat) → Nat.lbp p y m → (∀ (n : Nat), LT.lt n y → Membership.mem (p n …
          al : ∀ (n : Nat), LT.lt n m → Membership.mem (p n) Bool.false
          pm : (p m).Dom
          e : Eq ((p m).get pm) Bool.false
          n : Nat
          h : LE.le n m
          ⊢ Membership.mem (p n) Bool.false
        -/
        cases' h.lt_or_eq_dec with h h
          /-
            case false.inl
            p : PFun Nat Bool
            H : Exists fun n => And (Membership.mem (p n) Bool.true) (∀ (k : Nat), LT.lt k …
            m : Nat
            IH : (y : Nat) → Nat.lbp p y m → (∀ (n : Nat), LT.lt n y → Membership.mem (p n …
            al : ∀ (n : Nat), LT.lt n m → Membership.mem (p n) Bool.false
            pm : (p m).Dom
            e : Eq ((p m).get pm) Bool.false
            n : Nat
            h✝ : LE.le n m
            h : LT.lt n m
            ⊢ Membership.mem (p n) Bool.false
          -/
        · exact al _ h
          /-
            🎉 no goals
          -/
          /-
            case false.inr
            p : PFun Nat Bool
            H : Exists fun n => And (Membership.mem (p n) Bool.true) (∀ (k : Nat), LT.lt k …
            m : Nat
            IH : (y : Nat) → Nat.lbp p y m → (∀ (n : Nat), LT.lt n y → Membership.mem (p n …
            al : ∀ (n : Nat), LT.lt n m → Membership.mem (p n) Bool.false
            pm : (p m).Dom
            e : Eq ((p m).get pm) Bool.false
            n : Nat
            h✝ : LE.le n m
            h : Eq n m
            ⊢ Membership.mem (p n) Bool.false
          -/
        · rw [h]
          /-
            case false.inr
            p : PFun Nat Bool
            H : Exists fun n => And (Membership.mem (p n) Bool.true) (∀ (k : Nat), LT.lt k …
            m : Nat
            IH : (y : Nat) → Nat.lbp p y m → (∀ (n : Nat), LT.lt n y → Membership.mem (p n …
            al : ∀ (n : Nat), LT.lt n m → Membership.mem (p n) Bool.false
            pm : (p m).Dom
            e : Eq ((p m).get pm) Bool.false
            n : Nat
            h✝ : LE.le n m
            h : Eq n m
            ⊢ Membership.mem (p m) Bool.false
          -/
          exact ⟨_, e⟩
          /-
            🎉 no goals
          -/
        /-
          case true
          p : PFun Nat Bool
          H : Exists fun n => And (Membership.mem (p n) Bool.true) (∀ (k : Nat), LT.lt k …
          m : Nat
          IH : (y : Nat) → Nat.lbp p y m → (∀ (n : Nat), LT.lt n y → Membership.mem (p n …
          al : ∀ (n : Nat), LT.lt n m → Membership.mem (p n) Bool.false
          pm : (p m).Dom
          e : Eq ((p m).get pm) Bool.true
          ⊢ Subtype fun n => And (Membership.mem (p n) Bool.true) (∀ (m : Nat), LT.lt m  …
        -/
      · exact ⟨m, ⟨_, e⟩, al⟩)
        /-
          🎉 no goals
        -/


def rfind (p : ℕ →. Bool) : Part ℕ :=
  ⟨_, fun h => (rfindX p h).1⟩


theorem rfind_spec {p : ℕ →. Bool} {n : ℕ} (h : n ∈ rfind p) : true ∈ p n :=
  h.snd ▸ (rfindX p h.fst).2.1


theorem rfind_min {p : ℕ →. Bool} {n : ℕ} (h : n ∈ rfind p) : ∀ {m : ℕ}, m < n → false ∈ p m :=
  @(h.snd ▸ @((rfindX p h.fst).2.2))


@[simp]
theorem rfind_dom {p : ℕ →. Bool} :
    (rfind p).Dom ↔ ∃ n, true ∈ p n ∧ ∀ {m : ℕ}, m < n → (p m).Dom :=
  Iff.rfl


theorem rfind_dom' {p : ℕ →. Bool} :
    (rfind p).Dom ↔ ∃ n, true ∈ p n ∧ ∀ {m : ℕ}, m ≤ n → (p m).Dom :=
  exists_congr fun _ =>
    and_congr_right fun pn =>
      ⟨fun H _ h => (Decidable.eq_or_lt_of_le h).elim (fun e => e.symm ▸ pn.fst) (H _), fun H _ h =>
        H (le_of_lt h)⟩


@[simp]
theorem mem_rfind {p : ℕ →. Bool} {n : ℕ} :
    n ∈ rfind p ↔ true ∈ p n ∧ ∀ {m : ℕ}, m < n → false ∈ p m :=
  ⟨fun h => ⟨rfind_spec h, @rfind_min _ _ h⟩, fun ⟨h₁, h₂⟩ => by
    /-
      p : PFun Nat Bool
      n : Nat
      x✝ : And (Membership.mem (p n) Bool.true) (∀ {m : Nat}, LT.lt m n → Membership …
      h₁ : Membership.mem (p n) Bool.true
      h₂ : ∀ {m : Nat}, LT.lt m n → Membership.mem (p m) Bool.false
      ⊢ Membership.mem (Nat.rfind p) n
    -/
    let ⟨m, hm⟩ := dom_iff_mem.1 <| (@rfind_dom p).2 ⟨_, h₁, fun {m} mn => (h₂ mn).fst⟩
    /-
      p : PFun Nat Bool
      n : Nat
      x✝ : And (Membership.mem (p n) Bool.true) (∀ {m : Nat}, LT.lt m n → Membership …
      h₁ : Membership.mem (p n) Bool.true
      h₂ : ∀ {m : Nat}, LT.lt m n → Membership.mem (p m) Bool.false
      m : Nat
      hm : Membership.mem (Nat.rfind p) m
      ⊢ Membership.mem (Nat.rfind p) n
    -/
    rcases lt_trichotomy m n with (h | h | h)
      /-
        case inl
        p : PFun Nat Bool
        n : Nat
        x✝ : And (Membership.mem (p n) Bool.true) (∀ {m : Nat}, LT.lt m n → Membership …
        h₁ : Membership.mem (p n) Bool.true
        h₂ : ∀ {m : Nat}, LT.lt m n → Membership.mem (p m) Bool.false
        m : Nat
        hm : Membership.mem (Nat.rfind p) m
        h : LT.lt m n
        ⊢ Membership.mem (Nat.rfind p) n
      -/
    · injection mem_unique (h₂ h) (rfind_spec hm)
      /-
        🎉 no goals
      -/
      /-
        case inr.inl
        p : PFun Nat Bool
        n : Nat
        x✝ : And (Membership.mem (p n) Bool.true) (∀ {m : Nat}, LT.lt m n → Membership …
        h₁ : Membership.mem (p n) Bool.true
        h₂ : ∀ {m : Nat}, LT.lt m n → Membership.mem (p m) Bool.false
        m : Nat
        hm : Membership.mem (Nat.rfind p) m
        h : Eq m n
        ⊢ Membership.mem (Nat.rfind p) n
      -/
    · rwa [← h]
      /-
        🎉 no goals
      -/
      /-
        case inr.inr
        p : PFun Nat Bool
        n : Nat
        x✝ : And (Membership.mem (p n) Bool.true) (∀ {m : Nat}, LT.lt m n → Membership …
        h₁ : Membership.mem (p n) Bool.true
        h₂ : ∀ {m : Nat}, LT.lt m n → Membership.mem (p m) Bool.false
        m : Nat
        hm : Membership.mem (Nat.rfind p) m
        h : LT.lt n m
        ⊢ Membership.mem (Nat.rfind p) n
      -/
    · injection mem_unique h₁ (rfind_min hm h)⟩
      /-
        🎉 no goals
      -/


theorem rfind_min' {p : ℕ → Bool} {m : ℕ} (pm : p m) : ∃ n ∈ rfind p, n ≤ m :=
  have : true ∈ (p : ℕ →. Bool) m := ⟨trivial, pm⟩
  let ⟨n, hn⟩ := dom_iff_mem.1 <| (@rfind_dom p).2 ⟨m, this, fun {_} _ => ⟨⟩⟩
                               /-
                                 p : Nat → Bool
                                 m : Nat
                                 pm : Eq (p m) Bool.true
                                 this : Membership.mem (↑p m) Bool.true
                                 n : Nat
                                 hn : Membership.mem (Nat.rfind ↑p) n
                                 h : LT.lt m n
                                 ⊢ False
                               -/
  ⟨n, hn, not_lt.1 fun h => by injection mem_unique this (rfind_min hn h)⟩
                               /-
                                 🎉 no goals
                               -/


theorem rfind_zero_none (p : ℕ →. Bool) (p0 : p 0 = Part.none) : rfind p = Part.none :=
  eq_none_iff.2 fun _ h =>
    let ⟨_, _, h₂⟩ := rfind_dom'.1 h.fst
    (p0 ▸ h₂ (zero_le _) : (@Part.none Bool).Dom)


def rfindOpt {α} (f : ℕ → Option α) : Part α :=
  (rfind fun n => (f n).isSome).bind fun n => f n


theorem rfindOpt_spec {α} {f : ℕ → Option α} {a} (h : a ∈ rfindOpt f) : ∃ n, a ∈ f n :=
  let ⟨n, _, h₂⟩ := mem_bind_iff.1 h
  ⟨n, mem_coe.1 h₂⟩


theorem rfindOpt_dom {α} {f : ℕ → Option α} : (rfindOpt f).Dom ↔ ∃ n a, a ∈ f n :=
  ⟨fun h => (rfindOpt_spec ⟨h, rfl⟩).imp fun _ h => ⟨_, h⟩, fun h => by
    /-
      α : Type u_1
      f : Nat → Option α
      h : Exists fun n => Exists fun a => Membership.mem (f n) a
      ⊢ (Nat.rfindOpt f).Dom
    -/
    have h' : ∃ n, (f n).isSome := h.imp fun n => Option.isSome_iff_exists.2
    /-
      α : Type u_1
      f : Nat → Option α
      h : Exists fun n => Exists fun a => Membership.mem (f n) a
      h' : Exists fun n => Eq (f n).isSome Bool.true
      ⊢ (Nat.rfindOpt f).Dom
    -/
    have s := Nat.find_spec h'
    have fd : (rfind fun n => (f n).isSome).Dom :=
      ⟨Nat.find h', by simpa using s.symm, fun _ _ => trivial⟩
    /-
      α : Type u_1
      f : Nat → Option α
      h : Exists fun n => Exists fun a => Membership.mem (f n) a
      h' : Exists fun n => Eq (f n).isSome Bool.true
      s : Eq (f (Nat.find h')).isSome Bool.true
      fd : (Nat.rfind fun n => ↑(Option.some (f n).isSome)).Dom
      ⊢ (Nat.rfindOpt f).Dom
    -/
    refine ⟨fd, ?_⟩
    /-
      α : Type u_1
      f : Nat → Option α
      h : Exists fun n => Exists fun a => Membership.mem (f n) a
      h' : Exists fun n => Eq (f n).isSome Bool.true
      s : Eq (f (Nat.find h')).isSome Bool.true
      fd : (Nat.rfind fun n => ↑(Option.some (f n).isSome)).Dom
      ⊢ ((fun b => (fun n => ↑(f n)) ((Nat.rfind fun n => ↑(Option.some (f n).isSome …
    -/
    have := rfind_spec (get_mem fd)
    /-
      α : Type u_1
      f : Nat → Option α
      h : Exists fun n => Exists fun a => Membership.mem (f n) a
      h' : Exists fun n => Eq (f n).isSome Bool.true
      s : Eq (f (Nat.find h')).isSome Bool.true
      fd : (Nat.rfind fun n => ↑(Option.some (f n).isSome)).Dom
      this : Membership.mem (↑(Option.some (f ((Nat.rfind fun n => ↑(Option.some (f  …
      ⊢ ((fun b => (fun n => ↑(f n)) ((Nat.rfind fun n => ↑(Option.some (f n).isSome …
    -/
    simpa using this⟩
    /-
      🎉 no goals
    -/


theorem rfindOpt_mono {α} {f : ℕ → Option α} (H : ∀ {a m n}, m ≤ n → a ∈ f m → a ∈ f n) {a} :
    a ∈ rfindOpt f ↔ ∃ n, a ∈ f n :=
  ⟨rfindOpt_spec, fun ⟨n, h⟩ => by
    /-
      α : Type u_1
      f : Nat → Option α
      H : ∀ {a : α} {m n : Nat}, LE.le m n → Membership.mem (f m) a → Membership.mem …
      a : α
      x✝ : Exists fun n => Membership.mem (f n) a
      n : Nat
      h : Membership.mem (f n) a
      ⊢ Membership.mem (Nat.rfindOpt f) a
    -/
    have h' := rfindOpt_dom.2 ⟨_, _, h⟩
    /-
      α : Type u_1
      f : Nat → Option α
      H : ∀ {a : α} {m n : Nat}, LE.le m n → Membership.mem (f m) a → Membership.mem …
      a : α
      x✝ : Exists fun n => Membership.mem (f n) a
      n : Nat
      h : Membership.mem (f n) a
      h' : (Nat.rfindOpt f).Dom
      ⊢ Membership.mem (Nat.rfindOpt f) a
    -/
    cases' rfindOpt_spec ⟨h', rfl⟩ with k hk
    /-
      case intro
      α : Type u_1
      f : Nat → Option α
      H : ∀ {a : α} {m n : Nat}, LE.le m n → Membership.mem (f m) a → Membership.mem …
      a : α
      x✝ : Exists fun n => Membership.mem (f n) a
      n : Nat
      h : Membership.mem (f n) a
      h' : (Nat.rfindOpt f).Dom
      k : Nat
      hk : Membership.mem (f k) ((Nat.rfindOpt f).get h')
      ⊢ Membership.mem (Nat.rfindOpt f) a
    -/
    have := (H (le_max_left _ _) h).symm.trans (H (le_max_right _ _) hk)
    /-
      case intro
      α : Type u_1
      f : Nat → Option α
      H : ∀ {a : α} {m n : Nat}, LE.le m n → Membership.mem (f m) a → Membership.mem …
      a : α
      x✝ : Exists fun n => Membership.mem (f n) a
      n : Nat
      h : Membership.mem (f n) a
      h' : (Nat.rfindOpt f).Dom
      k : Nat
      hk : Membership.mem (f k) ((Nat.rfindOpt f).get h')
      this : Eq (Option.some a) (Option.some ((Nat.rfindOpt f).get h'))
      ⊢ Membership.mem (Nat.rfindOpt f) a
    -/
    simp at this; simp [this, get_mem]⟩
                  /-
                    🎉 no goals
                  -/


/-- `PartRec f` means that the partial function `f : ℕ → ℕ` is partially recursive. -/
inductive Partrec : (ℕ →. ℕ) → Prop
  | zero : Partrec (pure 0)
  | succ : Partrec succ
  | left : Partrec ↑fun n : ℕ => n.unpair.1
  | right : Partrec ↑fun n : ℕ => n.unpair.2
  | pair {f g} : Partrec f → Partrec g → Partrec fun n => pair <$> f n <*> g n
  | comp {f g} : Partrec f → Partrec g → Partrec fun n => g n >>= f
  | prec {f g} : Partrec f → Partrec g → Partrec (unpaired fun a n =>
      n.rec (f a) fun y IH => do let i ← IH; g (pair a (pair y i)))
  | rfind {f} : Partrec f → Partrec fun a => rfind fun n => (fun m => m = 0) <$> f (pair a n)


theorem of_eq {f g : ℕ →. ℕ} (hf : Partrec f) (H : ∀ n, f n = g n) : Partrec g :=
  (funext H : f = g) ▸ hf


theorem of_eq_tot {f : ℕ →. ℕ} {g : ℕ → ℕ} (hf : Partrec f) (H : ∀ n, g n ∈ f n) : Partrec g :=
  hf.of_eq fun n => eq_some_iff.2 (H n)


theorem of_primrec {f : ℕ → ℕ} (hf : Nat.Primrec f) : Partrec f := by
  induction hf with
  | zero => exact zero
  | succ => exact succ
  | left => exact left
  | right => exact right
  | pair _ _ pf pg =>
    refine (pf.pair pg).of_eq_tot fun n => ?_
    simp [Seq.seq]
  | comp _ _ pf pg =>
    refine (pf.comp pg).of_eq_tot fun n => (by simp)
  | prec _ _ pf pg =>
    refine (pf.prec pg).of_eq_tot fun n => ?_
    simp only [unpaired, PFun.coe_val, bind_eq_bind]
    induction n.unpair.2 with
    | zero => simp
    | succ m IH =>
      simp only [mem_bind_iff, mem_some_iff]
      exact ⟨_, IH, rfl⟩


protected theorem some : Partrec some :=
  of_primrec Primrec.id


theorem none : Partrec fun _ => none :=
  (of_primrec (Nat.Primrec.const 1)).rfind.of_eq fun _ =>
                                     /-
                                       x✝² x✝¹ : Nat
                                       x✝ : Membership.mem (Nat.rfind fun n => Functor.map (fun m => Decidable.decide …
                                       h : (Nat.rfind fun n => Functor.map (fun m => Decidable.decide (Eq m 0)) ((↑fu …
                                       h✝ : Eq ((Nat.rfind fun n => Functor.map (fun m => Decidable.decide (Eq m 0))  …
                                       ⊢ False
                                     -/
    eq_none_iff.2 fun _ ⟨h, _⟩ => by simp at h
                                     /-
                                       🎉 no goals
                                     -/


theorem prec' {f g h} (hf : Partrec f) (hg : Partrec g) (hh : Partrec h) :
    Partrec fun a => (f a).bind fun n => n.rec (g a)
      fun y IH => do {let i ← IH; h (Nat.pair a (Nat.pair y i))} :=
  ((prec hg hh).comp (pair Partrec.some hf)).of_eq fun a =>
                    /-
                      f g h : PFun Nat Nat
                      hf : Nat.Partrec f
                      hg : Nat.Partrec g
                      hh : Nat.Partrec h
                      a s : Nat
                      ⊢ Iff (Membership.mem (Bind.bind (Seq.seq (Functor.map Nat.pair (Part.some a)) …
                    -/
    ext fun s => by simp [Seq.seq]
                    /-
                      🎉 no goals
                    -/


theorem ppred : Partrec fun n => ppred n :=
  have : Primrec₂ fun n m => if n = Nat.succ m then 0 else 1 :=
    (Primrec.ite
      (@PrimrecRel.comp _ _ _ _ _ _ _ _ _ _
        Primrec.eq Primrec.fst (_root_.Primrec.succ.comp Primrec.snd))
      (_root_.Primrec.const 0) (_root_.Primrec.const 1)).to₂
  (of_primrec (Primrec₂.unpaired'.2 this)).rfind.of_eq fun n => by
    /-
      this : Primrec₂ fun n m => ite (Eq n m.succ) 0 1
      n : Nat
      ⊢ Eq (Nat.rfind fun n_1 => Functor.map (fun m => Decidable.decide (Eq m 0)) (↑ …
    -/
    cases n <;> simp
    · exact
        eq_none_iff.2 fun a ⟨⟨m, h, _⟩, _⟩ => by
          simp [show 0 ≠ m.succ by intro h; injection h] at h
      /-
        case succ
        this : Primrec₂ fun n m => ite (Eq n m.succ) 0 1
        n✝ : Nat
        ⊢ Eq (Nat.rfind fun n => Part.some (Decidable.decide (Eq n✝ n))) (Part.some n✝)
      -/
    · refine eq_some_iff.2 ?_
      simp only [mem_rfind, not_true, IsEmpty.forall_iff, decide_true, mem_some_iff,
        false_eq_decide_iff, true_and]
      /-
        case succ
        this : Primrec₂ fun n m => ite (Eq n m.succ) 0 1
        n✝ : Nat
        ⊢ ∀ {m : Nat}, LT.lt m n✝ → Not (Eq n✝ m)
      -/
      intro m h
      /-
        case succ
        this : Primrec₂ fun n m => ite (Eq n m.succ) 0 1
        n✝ m : Nat
        h : LT.lt m n✝
        ⊢ Not (Eq n✝ m)
      -/
      simp [ne_of_gt h]
      /-
        🎉 no goals
      -/


/-- Partially recursive partial functions `α → σ` between `Primcodable` types -/
def Partrec {α σ} [Primcodable α] [Primcodable σ] (f : α →. σ) :=
  Nat.Partrec fun n => Part.bind (decode (α := α) n) fun a => (f a).map encode


/-- Partially recursive partial functions `α → β → σ` between `Primcodable` types -/
def Partrec₂ {α β σ} [Primcodable α] [Primcodable β] [Primcodable σ] (f : α → β →. σ) :=
  Partrec fun p : α × β => f p.1 p.2


/-- Computable functions `α → σ` between `Primcodable` types:
  a function is computable if and only if it is partially recursive (as a partial function) -/
def Computable {α σ} [Primcodable α] [Primcodable σ] (f : α → σ) :=
  Partrec (f : α →. σ)


/-- Computable functions `α → β → σ` between `Primcodable` types -/
def Computable₂ {α β σ} [Primcodable α] [Primcodable β] [Primcodable σ] (f : α → β → σ) :=
  Computable fun p : α × β => f p.1 p.2


theorem Primrec.to_comp {α σ} [Primcodable α] [Primcodable σ] {f : α → σ} (hf : Primrec f) :
    Computable f :=
  (Nat.Partrec.ppred.comp (Nat.Partrec.of_primrec hf)).of_eq fun n => by
    /-
      α : Type u_1
      σ : Type u_2
      inst✝¹ : Primcodable α
      inst✝ : Primcodable σ
      f : α → σ
      hf : Primrec f
      n : Nat
      ⊢ Eq (Bind.bind ((↑fun n => Encodable.encode (Option.map f (Encodable.decode n …
    -/
                                      /-
                                        🎉 no goals
                                      -/
    simp; cases decode (α := α) n <;> simp
                                      /-
                                        🎉 no goals
                                      -/


nonrec theorem Primrec₂.to_comp {α β σ} [Primcodable α] [Primcodable β] [Primcodable σ]
    {f : α → β → σ} (hf : Primrec₂ f) : Computable₂ f :=
  hf.to_comp


protected theorem Computable.partrec {α σ} [Primcodable α] [Primcodable σ] {f : α → σ}
    (hf : Computable f) : Partrec (f : α →. σ) :=
  hf


protected theorem Computable₂.partrec₂ {α β σ} [Primcodable α] [Primcodable β] [Primcodable σ]
    {f : α → β → σ} (hf : Computable₂ f) : Partrec₂ fun a => (f a : β →. σ) :=
  hf


theorem of_eq {f g : α → σ} (hf : Computable f) (H : ∀ n, f n = g n) : Computable g :=
  (funext H : f = g) ▸ hf


theorem const (s : σ) : Computable fun _ : α => s :=
  (Primrec.const _).to_comp


theorem ofOption {f : α → Option β} (hf : Computable f) : Partrec fun a => (f a : Part β) :=
  (Nat.Partrec.ppred.comp hf).of_eq fun n => by
    /-
      α : Type u_1
      β : Type u_2
      inst✝¹ : Primcodable α
      inst✝ : Primcodable β
      f : α → Option β
      hf : Computable f
      n : Nat
      ⊢ Eq (Bind.bind ((↑(Encodable.decode n)).bind fun a => Part.map Encodable.enco …
    -/
                                        /-
                                          🎉 no goals
                                        -/
    cases' decode (α := α) n with a <;> simp
    /-
      case some
      α : Type u_1
      β : Type u_2
      inst✝¹ : Primcodable α
      inst✝ : Primcodable β
      f : α → Option β
      hf : Computable f
      n : Nat
      a : α
      ⊢ Eq (↑(Encodable.encode (f a)).ppred) (Part.map Encodable.encode ↑(f a))
    -/
                          /-
                            🎉 no goals
                          -/
    cases' f a with b <;> simp
                          /-
                            🎉 no goals
                          -/


theorem to₂ {f : α × β → σ} (hf : Computable f) : Computable₂ fun a b => f (a, b) :=
  hf.of_eq fun ⟨_, _⟩ => rfl


protected theorem id : Computable (@id α) :=
  Primrec.id.to_comp


theorem fst : Computable (@Prod.fst α β) :=
  Primrec.fst.to_comp


theorem snd : Computable (@Prod.snd α β) :=
  Primrec.snd.to_comp


nonrec theorem pair {f : α → β} {g : α → γ} (hf : Computable f) (hg : Computable g) :
    Computable fun a => (f a, g a) :=
                                 /-
                                   α : Type u_1
                                   β : Type u_2
                                   γ : Type u_3
                                   inst✝² : Primcodable α
                                   inst✝¹ : Primcodable β
                                   inst✝ : Primcodable γ
                                   f : α → β
                                   g : α → γ
                                   hf : Computable f
                                   hg : Computable g
                                   n : Nat
                                   ⊢ Eq (Seq.seq (Functor.map Nat.pair ((↑(Encodable.decode n)).bind fun a => Par …
                                 -/
                                                             /-
                                                               🎉 no goals
                                                             -/
  (hf.pair hg).of_eq fun n => by cases decode (α := α) n <;> simp [Seq.seq]
                                                             /-
                                                               🎉 no goals
                                                             -/


theorem unpair : Computable Nat.unpair :=
  Primrec.unpair.to_comp


theorem succ : Computable Nat.succ :=
  Primrec.succ.to_comp


theorem pred : Computable Nat.pred :=
  Primrec.pred.to_comp


theorem nat_bodd : Computable Nat.bodd :=
  Primrec.nat_bodd.to_comp


theorem nat_div2 : Computable Nat.div2 :=
  Primrec.nat_div2.to_comp


theorem sum_inl : Computable (@Sum.inl α β) :=
  Primrec.sum_inl.to_comp


theorem sum_inr : Computable (@Sum.inr α β) :=
  Primrec.sum_inr.to_comp


theorem list_cons : Computable₂ (@List.cons α) :=
  Primrec.list_cons.to_comp


theorem list_reverse : Computable (@List.reverse α) :=
  Primrec.list_reverse.to_comp


theorem list_get? : Computable₂ (@List.get? α) :=
  Primrec.list_get?.to_comp


theorem list_append : Computable₂ ((· ++ ·) : List α → List α → List α) :=
  Primrec.list_append.to_comp


theorem list_concat : Computable₂ fun l (a : α) => l ++ [a] :=
  Primrec.list_concat.to_comp


theorem list_length : Computable (@List.length α) :=
  Primrec.list_length.to_comp


theorem vector_cons {n} : Computable₂ (@List.Vector.cons α n) :=
  Primrec.vector_cons.to_comp


theorem vector_toList {n} : Computable (@List.Vector.toList α n) :=
  Primrec.vector_toList.to_comp


theorem vector_length {n} : Computable (@List.Vector.length α n) :=
  Primrec.vector_length.to_comp


theorem vector_head {n} : Computable (@List.Vector.head α n) :=
  Primrec.vector_head.to_comp


theorem vector_tail {n} : Computable (@List.Vector.tail α n) :=
  Primrec.vector_tail.to_comp


theorem vector_get {n} : Computable₂ (@List.Vector.get α n) :=
  Primrec.vector_get.to_comp


theorem vector_ofFn' {n} : Computable (@List.Vector.ofFn α n) :=
  Primrec.vector_ofFn'.to_comp


theorem fin_app {n} : Computable₂ (@id (Fin n → σ)) :=
  Primrec.fin_app.to_comp


protected theorem encode : Computable (@encode α _) :=
  Primrec.encode.to_comp


protected theorem decode : Computable (decode (α := α)) :=
  Primrec.decode.to_comp


protected theorem ofNat (α) [Denumerable α] : Computable (ofNat α) :=
  (Primrec.ofNat _).to_comp


theorem encode_iff {f : α → σ} : (Computable fun a => encode (f a)) ↔ Computable f :=
  Iff.rfl


theorem option_some : Computable (@Option.some α) :=
  Primrec.option_some.to_comp


theorem of_eq {f g : α →. σ} (hf : Partrec f) (H : ∀ n, f n = g n) : Partrec g :=
  (funext H : f = g) ▸ hf


theorem of_eq_tot {f : α →. σ} {g : α → σ} (hf : Partrec f) (H : ∀ n, g n ∈ f n) : Computable g :=
  hf.of_eq fun a => eq_some_iff.2 (H a)


theorem none : Partrec fun _ : α => @Part.none σ :=
                                     /-
                                       α : Type u_1
                                       σ : Type u_3
                                       inst✝¹ : Primcodable α
                                       inst✝ : Primcodable σ
                                       n : Nat
                                       ⊢ Eq Part.none ((↑(Encodable.decode n)).bind fun a => Part.map Encodable.encod …
                                     -/
                                                                 /-
                                                                   🎉 no goals
                                                                 -/
  Nat.Partrec.none.of_eq fun n => by cases decode (α := α) n <;> simp
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


protected theorem some : Partrec (@Part.some α) :=
  Computable.id


theorem _root_.Decidable.Partrec.const' (s : Part σ) [Decidable s.Dom] : Partrec fun _ : α => s :=
  (Computable.ofOption (const (toOption s))).of_eq fun _ => of_toOption s


theorem const' (s : Part σ) : Partrec fun _ : α => s :=
  haveI := Classical.dec s.Dom
  Decidable.Partrec.const' s


protected theorem bind {f : α →. β} {g : α → β →. σ} (hf : Partrec f) (hg : Partrec₂ g) :
    Partrec fun a => (f a).bind (g a) :=
  (hg.comp (Nat.Partrec.some.pair hf)).of_eq fun n => by
    /-
      α : Type u_1
      β : Type u_2
      σ : Type u_3
      inst✝² : Primcodable α
      inst✝¹ : Primcodable β
      inst✝ : Primcodable σ
      f : PFun α β
      g : α → PFun β σ
      hf : Partrec f
      hg : Partrec₂ g
      n : Nat
      ⊢ Eq (Bind.bind (Seq.seq (Functor.map Nat.pair (Part.some n)) fun x => (↑(Enco …
    -/
                                                            /-
                                                              🎉 no goals
                                                            -/
    simp [Seq.seq]; cases' e : decode (α := α) n with a <;> simp [e, encodek]
                                                            /-
                                                              🎉 no goals
                                                            -/


theorem map {f : α →. β} {g : α → β → σ} (hf : Partrec f) (hg : Computable₂ g) :
    Partrec fun a => (f a).map (g a) := by
  /-
    α : Type u_1
    β : Type u_2
    σ : Type u_3
    inst✝² : Primcodable α
    inst✝¹ : Primcodable β
    inst✝ : Primcodable σ
    f : PFun α β
    g : α → β → σ
    hf : Partrec f
    hg : Computable₂ g
    ⊢ Partrec fun a => Part.map (g a) (f a)
  -/
  simpa [bind_some_eq_map] using @Partrec.bind _ _ _ _ _ _ _ (fun a => Part.some ∘ (g a)) hf hg
  /-
    🎉 no goals
  -/


theorem to₂ {f : α × β →. σ} (hf : Partrec f) : Partrec₂ fun a b => f (a, b) :=
  hf.of_eq fun ⟨_, _⟩ => rfl


theorem nat_rec {f : α → ℕ} {g : α →. σ} {h : α → ℕ × σ →. σ} (hf : Computable f) (hg : Partrec g)
    (hh : Partrec₂ h) : Partrec fun a => (f a).rec (g a) fun y IH => IH.bind fun i => h a (y, i) :=
  (Nat.Partrec.prec' hf hg hh).of_eq fun n => by
    /-
      α : Type u_1
      σ : Type u_3
      inst✝¹ : Primcodable α
      inst✝ : Primcodable σ
      f : α → Nat
      g : PFun α σ
      h : α → PFun (Prod Nat σ) σ
      hf : Computable f
      hg : Partrec g
      hh : Partrec₂ h
      n : Nat
      ⊢ Eq (((↑(Encodable.decode n)).bind fun a => Part.map Encodable.encode (↑f a)) …
    -/
                                            /-
                                              🎉 no goals
                                            -/
    cases' e : decode (α := α) n with a <;> simp [e]
    /-
      case some
      α : Type u_1
      σ : Type u_3
      inst✝¹ : Primcodable α
      inst✝ : Primcodable σ
      f : α → Nat
      g : PFun α σ
      h : α → PFun (Prod Nat σ) σ
      hf : Computable f
      hg : Partrec g
      hh : Partrec₂ h
      n : Nat
      a : α
      e : Eq (Encodable.decode n) (Option.some a)
      ⊢ Eq (Nat.rec (Part.map Encodable.encode (g a)) (fun y IH => IH.bind fun i =>  …
    -/
                                 /-
                                   🎉 no goals
                                 -/
    induction' f a with m IH <;> simp
    /-
      case some.succ
      α : Type u_1
      σ : Type u_3
      inst✝¹ : Primcodable α
      inst✝ : Primcodable σ
      f : α → Nat
      g : PFun α σ
      h : α → PFun (Prod Nat σ) σ
      hf : Computable f
      hg : Partrec g
      hh : Partrec₂ h
      n : Nat
      a : α
      e : Eq (Encodable.decode n) (Option.some a)
      m : Nat
      IH : Eq (Nat.rec (Part.map Encodable.encode (g a)) (fun y IH => IH.bind fun i  …
      ⊢ Eq ((Nat.rec (Part.map Encodable.encode (g a)) (fun y IH => IH.bind fun i => …
    -/
    rw [IH, Part.bind_map]
    /-
      case some.succ
      α : Type u_1
      σ : Type u_3
      inst✝¹ : Primcodable α
      inst✝ : Primcodable σ
      f : α → Nat
      g : PFun α σ
      h : α → PFun (Prod Nat σ) σ
      hf : Computable f
      hg : Partrec g
      hh : Partrec₂ h
      n : Nat
      a : α
      e : Eq (Encodable.decode n) (Option.some a)
      m : Nat
      IH : Eq (Nat.rec (Part.map Encodable.encode (g a)) (fun y IH => IH.bind fun i  …
      ⊢ Eq ((Nat.rec (g a) (fun y IH => IH.bind fun i => h a { fst := y, snd := i }) …
    -/
    congr; funext s
    /-
      case some.succ.e_g.h
      α : Type u_1
      σ : Type u_3
      inst✝¹ : Primcodable α
      inst✝ : Primcodable σ
      f : α → Nat
      g : PFun α σ
      h : α → PFun (Prod Nat σ) σ
      hf : Computable f
      hg : Partrec g
      hh : Partrec₂ h
      n : Nat
      a : α
      e : Eq (Encodable.decode n) (Option.some a)
      m : Nat
      IH : Eq (Nat.rec (Part.map Encodable.encode (g a)) (fun y IH => IH.bind fun i  …
      s : σ
      ⊢ Eq ((↑(Option.map (Function.comp (Prod.mk a) (Prod.mk m)) (Encodable.decode  …
    -/
    simp [encodek]
    /-
      🎉 no goals
    -/


nonrec theorem comp {f : β →. σ} {g : α → β} (hf : Partrec f) (hg : Computable g) :
    Partrec fun a => f (g a) :=
                                 /-
                                   α : Type u_1
                                   β : Type u_2
                                   σ : Type u_3
                                   inst✝² : Primcodable α
                                   inst✝¹ : Primcodable β
                                   inst✝ : Primcodable σ
                                   f : PFun β σ
                                   g : α → β
                                   hf : Partrec f
                                   hg : Computable g
                                   n : Nat
                                   ⊢ Eq (Bind.bind ((↑(Encodable.decode n)).bind fun a => Part.map Encodable.enco …
                                 -/
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/
  (hf.comp hg).of_eq fun n => by simp; cases' e : decode (α := α) n with a <;> simp [e, encodek]
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/


                                                               /-
                                                                 f : PFun Nat Nat
                                                                 ⊢ Iff (Partrec f) (Nat.Partrec f)
                                                               -/
theorem nat_iff {f : ℕ →. ℕ} : Partrec f ↔ Nat.Partrec f := by simp [Partrec, map_id']
                                                               /-
                                                                 🎉 no goals
                                                               -/


theorem map_encode_iff {f : α →. σ} : (Partrec fun a => (f a).map encode) ↔ Partrec f :=
  Iff.rfl


theorem unpaired {f : ℕ → ℕ →. α} : Partrec (Nat.unpaired f) ↔ Partrec₂ f :=
               /-
                 α : Type u_1
                 inst✝ : Primcodable α
                 f : Nat → PFun Nat α
                 h : Partrec (Nat.unpaired f)
                 ⊢ Partrec₂ f
               -/
  ⟨fun h => by simpa using Partrec.comp (g := fun p : ℕ × ℕ => (p.1, p.2)) h Primrec₂.pair.to_comp,
               /-
                 🎉 no goals
               -/
    fun h => h.comp Primrec.unpair.to_comp⟩


theorem unpaired' {f : ℕ → ℕ →. ℕ} : Nat.Partrec (Nat.unpaired f) ↔ Partrec₂ f :=
  Partrec.nat_iff.symm.trans unpaired


nonrec theorem comp {f : β → γ →. σ} {g : α → β} {h : α → γ} (hf : Partrec₂ f) (hg : Computable g)
    (hh : Computable h) : Partrec fun a => f (g a) (h a) :=
  hf.comp (hg.pair hh)


theorem comp₂ {f : γ → δ →. σ} {g : α → β → γ} {h : α → β → δ} (hf : Partrec₂ f)
    (hg : Computable₂ g) (hh : Computable₂ h) : Partrec₂ fun a b => f (g a b) (h a b) :=
  hf.comp hg hh


nonrec theorem comp {f : β → σ} {g : α → β} (hf : Computable f) (hg : Computable g) :
    Computable fun a => f (g a) :=
  hf.comp hg


theorem comp₂ {f : γ → σ} {g : α → β → γ} (hf : Computable f) (hg : Computable₂ g) :
    Computable₂ fun a b => f (g a b) :=
  hf.comp hg


theorem mk {f : α → β → σ} (hf : Computable fun p : α × β => f p.1 p.2) : Computable₂ f := hf


nonrec theorem comp {f : β → γ → σ} {g : α → β} {h : α → γ} (hf : Computable₂ f)
    (hg : Computable g) (hh : Computable h) : Computable fun a => f (g a) (h a) :=
  hf.comp (hg.pair hh)


theorem comp₂ {f : γ → δ → σ} {g : α → β → γ} {h : α → β → δ} (hf : Computable₂ f)
    (hg : Computable₂ g) (hh : Computable₂ h) : Computable₂ fun a b => f (g a b) (h a b) :=
  hf.comp hg hh


theorem rfind {p : α → ℕ →. Bool} (hp : Partrec₂ p) : Partrec fun a => Nat.rfind (p a) :=
  (Nat.Partrec.rfind <|
        hp.map ((Primrec.dom_bool fun b => cond b 0 1).comp Primrec.snd).to₂.to_comp).of_eq
    fun n => by
    /-
      α : Type u_1
      inst✝ : Primcodable α
      p : α → PFun Nat Bool
      hp : Partrec₂ p
      n : Nat
      ⊢ Eq (Nat.rfind fun n_1 => Functor.map (fun m => Decidable.decide (Eq m 0)) (( …
    -/
                                            /-
                                              🎉 no goals
                                            -/
    cases' e : decode (α := α) n with a <;> simp [e, Nat.rfind_zero_none, map_id']
    /-
      case some
      α : Type u_1
      inst✝ : Primcodable α
      p : α → PFun Nat Bool
      hp : Partrec₂ p
      n : Nat
      a : α
      e : Eq (Encodable.decode n) (Option.some a)
      ⊢ Eq (Nat.rfind fun n => Part.map (fun m => Decidable.decide (Eq m 0)) (Part.m …
    -/
    congr; funext n
    /-
      case some.e_p.h
      α : Type u_1
      inst✝ : Primcodable α
      p : α → PFun Nat Bool
      hp : Partrec₂ p
      n✝ : Nat
      a : α
      e : Eq (Encodable.decode n✝) (Option.some a)
      n : Nat
      ⊢ Eq (Part.map (fun m => Decidable.decide (Eq m 0)) (Part.map (fun b => cond b …
    -/
    simp only [map_map, Function.comp]
    /-
      case some.e_p.h
      α : Type u_1
      inst✝ : Primcodable α
      p : α → PFun Nat Bool
      hp : Partrec₂ p
      n✝ : Nat
      a : α
      e : Eq (Encodable.decode n✝) (Option.some a)
      n : Nat
      ⊢ Eq (Part.map (Function.comp (fun m => Decidable.decide (Eq m 0)) fun b => co …
    -/
    refine map_id' (fun b => ?_) _
    /-
      case some.e_p.h
      α : Type u_1
      inst✝ : Primcodable α
      p : α → PFun Nat Bool
      hp : Partrec₂ p
      n✝ : Nat
      a : α
      e : Eq (Encodable.decode n✝) (Option.some a)
      n : Nat
      b : Bool
      ⊢ Eq (Function.comp (fun m => Decidable.decide (Eq m 0)) (fun b => cond b 0 1) …
    -/
                /-
                  🎉 no goals
                -/
    cases b <;> rfl
                /-
                  🎉 no goals
                -/


theorem rfindOpt {f : α → ℕ → Option σ} (hf : Computable₂ f) :
    Partrec fun a => Nat.rfindOpt (f a) :=
  (rfind (Primrec.option_isSome.to_comp.comp hf).partrec.to₂).bind (ofOption hf)


theorem nat_casesOn_right {f : α → ℕ} {g : α → σ} {h : α → ℕ →. σ} (hf : Computable f)
    (hg : Computable g) (hh : Partrec₂ h) : Partrec fun a => (f a).casesOn (some (g a)) (h a) :=
  (nat_rec hf hg (hh.comp fst (pred.comp <| hf.comp fst)).to₂).of_eq fun a => by
    /-
      α : Type u_1
      σ : Type u_2
      inst✝¹ : Primcodable α
      inst✝ : Primcodable σ
      f : α → Nat
      g : α → σ
      h : α → PFun Nat σ
      hf : Computable f
      hg : Computable g
      hh : Partrec₂ h
      a : α
      ⊢ Eq (Nat.rec (↑g a) (fun y IH => IH.bind fun i => h { fst := a, snd := { fst  …
    -/
                                                                         /-
                                                                           🎉 no goals
                                                                         -/
    simp only [PFun.coe_val, Nat.pred_eq_sub_one]; cases' f a with n <;> simp
    /-
      case succ
      α : Type u_1
      σ : Type u_2
      inst✝¹ : Primcodable α
      inst✝ : Primcodable σ
      f : α → Nat
      g : α → σ
      h : α → PFun Nat σ
      hf : Computable f
      hg : Computable g
      hh : Partrec₂ h
      a : α
      n : Nat
      ⊢ Eq ((Nat.rec (Part.some (g a)) (fun y IH => IH.bind fun i => h a n) n).bind  …
    -/
    refine ext fun b => ⟨fun H => ?_, fun H => ?_⟩
      /-
        case succ.refine_1
        α : Type u_1
        σ : Type u_2
        inst✝¹ : Primcodable α
        inst✝ : Primcodable σ
        f : α → Nat
        g : α → σ
        h : α → PFun Nat σ
        hf : Computable f
        hg : Computable g
        hh : Partrec₂ h
        a : α
        n : Nat
        b : σ
        H : Membership.mem ((Nat.rec (Part.some (g a)) (fun y IH => IH.bind fun i => h …
        ⊢ Membership.mem (h a n) b
      -/
    · rcases mem_bind_iff.1 H with ⟨c, _, h₂⟩
      /-
        case succ.refine_1.intro.intro
        α : Type u_1
        σ : Type u_2
        inst✝¹ : Primcodable α
        inst✝ : Primcodable σ
        f : α → Nat
        g : α → σ
        h : α → PFun Nat σ
        hf : Computable f
        hg : Computable g
        hh : Partrec₂ h
        a : α
        n : Nat
        b : σ
        H : Membership.mem ((Nat.rec (Part.some (g a)) (fun y IH => IH.bind fun i => h …
        c : σ
        left✝ : Membership.mem (Nat.rec (Part.some (g a)) (fun y IH => IH.bind fun i = …
        h₂ : Membership.mem (h a n) b
        ⊢ Membership.mem (h a n) b
      -/
      exact h₂
      /-
        🎉 no goals
      -/
    · have : ∀ m, (Nat.rec (motive := fun _ => Part σ)
          (Part.some (g a)) (fun y IH => IH.bind fun _ => h a n) m).Dom := by
        intro m
        induction m <;> simp [*, H.fst]
      /-
        case succ.refine_2
        α : Type u_1
        σ : Type u_2
        inst✝¹ : Primcodable α
        inst✝ : Primcodable σ
        f : α → Nat
        g : α → σ
        h : α → PFun Nat σ
        hf : Computable f
        hg : Computable g
        hh : Partrec₂ h
        a : α
        n : Nat
        b : σ
        H : Membership.mem (h a n) b
        this : ∀ (m : Nat), (Nat.rec (Part.some (g a)) (fun y IH => Part.bind IH fun x …
        ⊢ Membership.mem ((Nat.rec (Part.some (g a)) (fun y IH => IH.bind fun i => h a …
      -/
      exact ⟨⟨this n, H.fst⟩, H.snd⟩
      /-
        🎉 no goals
      -/


theorem bind_decode₂_iff {f : α →. σ} :
    Partrec f ↔ Nat.Partrec fun n => Part.bind (decode₂ α n) fun a => (f a).map encode :=
  ⟨fun hf =>
    nat_iff.1 <|
      (Computable.ofOption Primrec.decode₂.to_comp).bind <|
        (map hf (Computable.encode.comp snd).to₂).comp snd,
    fun h =>
                           /-
                             α : Type u_1
                             σ : Type u_2
                             inst✝¹ : Primcodable α
                             inst✝ : Primcodable σ
                             f : PFun α σ
                             h : Nat.Partrec fun n => (↑(Encodable.decode₂ α n)).bind fun a => Part.map Enc …
                             ⊢ Partrec fun a => Part.map Encodable.encode (f a)
                           -/
    map_encode_iff.1 <| by simpa [encodek₂] using (nat_iff.2 h).comp (@Computable.encode α _)⟩
                           /-
                             🎉 no goals
                           -/


theorem vector_mOfFn :
    ∀ {n} {f : Fin n → α →. σ},
      (∀ i, Partrec (f i)) → Partrec fun a : α => Vector.mOfFn fun i => f i a
  | 0, _, _ => const _
  | n + 1, f, hf => by
    /-
      α : Type u_1
      σ : Type u_2
      inst✝¹ : Primcodable α
      inst✝ : Primcodable σ
      n : Nat
      f : Fin (HAdd.hAdd n 1) → PFun α σ
      hf : ∀ (i : Fin (HAdd.hAdd n 1)), Partrec (f i)
      ⊢ Partrec fun a => List.Vector.mOfFn fun i => f i a
    -/
    simp only [Vector.mOfFn, Nat.add_eq, Nat.add_zero, pure_eq_some, bind_eq_bind]
    exact
      (hf 0).bind
        (Partrec.bind ((vector_mOfFn fun i => hf i.succ).comp fst)
          (Primrec.vector_cons.to_comp.comp (snd.comp fst) snd))


@[simp]
theorem Vector.mOfFn_part_some {α n} :
    ∀ f : Fin n → α,
      (List.Vector.mOfFn fun i => Part.some (f i)) = Part.some (List.Vector.ofFn f) :=
  Vector.mOfFn_pure


theorem option_some_iff {f : α → σ} : (Computable fun a => Option.some (f a)) ↔ Computable f :=
  ⟨fun h => encode_iff.1 <| Primrec.pred.to_comp.comp <| encode_iff.2 h, option_some.comp⟩


theorem bind_decode_iff {f : α → β → Option σ} :
    (Computable₂ fun a n => (decode (α := β) n).bind (f a)) ↔ Computable₂ f :=
  ⟨fun hf =>
    Nat.Partrec.of_eq
      (((Partrec.nat_iff.2
        (Nat.Partrec.ppred.comp <| Nat.Partrec.of_primrec <| Primcodable.prim (α := β))).comp
            snd).bind
        (Computable.comp hf fst).to₂.partrec₂)
      fun n => by
        simp only [decode_prod_val, decode_nat, Option.map_some', PFun.coe_val, bind_eq_bind,
          bind_some, Part.map_bind, map_some]
        /-
          α : Type u_1
          β : Type u_2
          σ : Type u_4
          inst✝² : Primcodable α
          inst✝¹ : Primcodable β
          inst✝ : Primcodable σ
          f : α → β → Option σ
          hf : Computable₂ fun a n => (Encodable.decode n).bind (f a)
          n : Nat
          ⊢ Eq ((↑((Encodable.decode (Nat.unpair n).1).bind fun a => Option.some { fst : …
        -/
                                             /-
                                               🎉 no goals
                                             -/
        cases decode (α := α) n.unpair.1 <;> simp
        /-
          case some
          α : Type u_1
          β : Type u_2
          σ : Type u_4
          inst✝² : Primcodable α
          inst✝¹ : Primcodable β
          inst✝ : Primcodable σ
          f : α → β → Option σ
          hf : Computable₂ fun a n => (Encodable.decode n).bind (f a)
          n : Nat
          val✝ : α
          ⊢ Eq ((↑(Encodable.encode (Encodable.decode (Nat.unpair n).2)).ppred).bind fun …
        -/
                                             /-
                                               🎉 no goals
                                             -/
        cases decode (α := β) n.unpair.2 <;> simp,
                                             /-
                                               🎉 no goals
                                             -/
    fun hf => by
    have :
      Partrec fun a : α × ℕ =>
        (encode (decode (α := β) a.2)).casesOn (some Option.none)
          fun n => Part.map (f a.1) (decode (α := β) n) :=
      Partrec.nat_casesOn_right
        (h := fun (a : α × ℕ) (n : ℕ) ↦ map (fun b ↦ f a.1 b) (Part.ofOption (decode n)))
        (Primrec.encdec.to_comp.comp snd) (const Option.none)
        ((ofOption (Computable.decode.comp snd)).map (hf.comp (fst.comp <| fst.comp fst) snd).to₂)
    /-
      α : Type u_1
      β : Type u_2
      σ : Type u_4
      inst✝² : Primcodable α
      inst✝¹ : Primcodable β
      inst✝ : Primcodable σ
      f : α → β → Option σ
      hf : Computable₂ f
      this : Partrec fun a => Nat.casesOn (Encodable.encode (Encodable.decode a.2))  …
      ⊢ Computable₂ fun a n => (Encodable.decode n).bind (f a)
    -/
    refine this.of_eq fun a => ?_
    /-
      α : Type u_1
      β : Type u_2
      σ : Type u_4
      inst✝² : Primcodable α
      inst✝¹ : Primcodable β
      inst✝ : Primcodable σ
      f : α → β → Option σ
      hf : Computable₂ f
      this : Partrec fun a => Nat.casesOn (Encodable.encode (Encodable.decode a.2))  …
      a : Prod α Nat
      ⊢ Eq (Nat.casesOn (Encodable.encode (Encodable.decode a.2)) (Part.some Option. …
    -/
                                        /-
                                          🎉 no goals
                                        -/
    simp; cases decode (α := β) a.2 <;> simp [encodek]⟩
                                        /-
                                          🎉 no goals
                                        -/


theorem map_decode_iff {f : α → β → σ} :
    (Computable₂ fun a n => (decode (α := β) n).map (f a)) ↔ Computable₂ f := by
  /-
    α : Type u_1
    β : Type u_2
    σ : Type u_4
    inst✝² : Primcodable α
    inst✝¹ : Primcodable β
    inst✝ : Primcodable σ
    f : α → β → σ
    ⊢ Iff (Computable₂ fun a n => Option.map (f a) (Encodable.decode n)) (Computab …
  -/
  convert (bind_decode_iff (f := fun a => Option.some ∘ f a)).trans option_some_iff
  /-
    case h.e'_1.h.e'_7.h.h
    α : Type u_1
    β : Type u_2
    σ : Type u_4
    inst✝² : Primcodable α
    inst✝¹ : Primcodable β
    inst✝ : Primcodable σ
    f : α → β → σ
    x✝¹ : α
    x✝ : Nat
    ⊢ Eq (Option.map (f x✝¹) (Encodable.decode x✝)) ((Encodable.decode x✝).bind (F …
  -/
  apply Option.map_eq_bind
  /-
    🎉 no goals
  -/


theorem nat_rec {f : α → ℕ} {g : α → σ} {h : α → ℕ × σ → σ} (hf : Computable f) (hg : Computable g)
    (hh : Computable₂ h) :
    Computable fun a => Nat.rec (motive := fun _ => σ) (g a) (fun y IH => h a (y, IH)) (f a) :=
                                                        /-
                                                          α : Type u_1
                                                          σ : Type u_4
                                                          inst✝¹ : Primcodable α
                                                          inst✝ : Primcodable σ
                                                          f : α → Nat
                                                          g : α → σ
                                                          h : α → Prod Nat σ → σ
                                                          hf : Computable f
                                                          hg : Computable g
                                                          hh : Computable₂ h
                                                          a : α
                                                          ⊢ Eq (Nat.rec (↑g a) (fun y IH => IH.bind fun i => ↑(h a) { fst := y, snd := i …
                                                        -/
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/
  (Partrec.nat_rec hf hg hh.partrec₂).of_eq fun a => by simp; induction f a <;> simp [*]
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/


theorem nat_casesOn {f : α → ℕ} {g : α → σ} {h : α → ℕ → σ} (hf : Computable f) (hg : Computable g)
    (hh : Computable₂ h) :
    Computable fun a => Nat.casesOn (motive := fun _ => σ) (f a) (g a) (h a) :=
  nat_rec hf hg (hh.comp fst <| fst.comp snd).to₂


theorem cond {c : α → Bool} {f : α → σ} {g : α → σ} (hc : Computable c) (hf : Computable f)
    (hg : Computable g) : Computable fun a => cond (c a) (f a) (g a) :=
                                                                         /-
                                                                           α : Type u_1
                                                                           σ : Type u_4
                                                                           inst✝¹ : Primcodable α
                                                                           inst✝ : Primcodable σ
                                                                           c : α → Bool
                                                                           f g : α → σ
                                                                           hc : Computable c
                                                                           hf : Computable f
                                                                           hg : Computable g
                                                                           a : α
                                                                           ⊢ Eq (Nat.casesOn (Encodable.encode (c a)) (g a) fun b => f { fst := a, snd := …
                                                                         -/
                                                                                       /-
                                                                                         🎉 no goals
                                                                                       -/
  (nat_casesOn (encode_iff.2 hc) hg (hf.comp fst).to₂).of_eq fun a => by cases c a <;> rfl
                                                                                       /-
                                                                                         🎉 no goals
                                                                                       -/


theorem option_casesOn {o : α → Option β} {f : α → σ} {g : α → β → σ} (ho : Computable o)
    (hf : Computable f) (hg : Computable₂ g) :
    @Computable _ σ _ _ fun a => Option.casesOn (o a) (f a) (g a) :=
  option_some_iff.1 <|
    (nat_casesOn (encode_iff.2 ho) (option_some_iff.2 hf) (map_decode_iff.2 hg)).of_eq fun a => by
      /-
        α : Type u_1
        β : Type u_2
        σ : Type u_4
        inst✝² : Primcodable α
        inst✝¹ : Primcodable β
        inst✝ : Primcodable σ
        o : α → Option β
        f : α → σ
        g : α → β → σ
        ho : Computable o
        hf : Computable f
        hg : Computable₂ g
        a : α
        ⊢ Eq (Nat.casesOn (Encodable.encode (o a)) (Option.some (f a)) fun n => Option …
      -/
                    /-
                      🎉 no goals
                    -/
      cases o a <;> simp [encodek]
                    /-
                      🎉 no goals
                    -/


theorem option_bind {f : α → Option β} {g : α → β → Option σ} (hf : Computable f)
    (hg : Computable₂ g) : Computable fun a => (f a).bind (g a) :=
                                                               /-
                                                                 α : Type u_1
                                                                 β : Type u_2
                                                                 σ : Type u_4
                                                                 inst✝² : Primcodable α
                                                                 inst✝¹ : Primcodable β
                                                                 inst✝ : Primcodable σ
                                                                 f : α → Option β
                                                                 g : α → β → Option σ
                                                                 hf : Computable f
                                                                 hg : Computable₂ g
                                                                 a : α
                                                                 ⊢ Eq (Option.casesOn (f a) Option.none (g a)) ((f a).bind (g a))
                                                               -/
                                                                             /-
                                                                               🎉 no goals
                                                                             -/
  (option_casesOn hf (const Option.none) hg).of_eq fun a => by cases f a <;> rfl
                                                                             /-
                                                                               🎉 no goals
                                                                             -/


theorem option_map {f : α → Option β} {g : α → β → σ} (hf : Computable f) (hg : Computable₂ g) :
    Computable fun a => (f a).map (g a) := by
  /-
    α : Type u_1
    β : Type u_2
    σ : Type u_4
    inst✝² : Primcodable α
    inst✝¹ : Primcodable β
    inst✝ : Primcodable σ
    f : α → Option β
    g : α → β → σ
    hf : Computable f
    hg : Computable₂ g
    ⊢ Computable fun a => Option.map (g a) (f a)
  -/
  convert option_bind hf (option_some.comp₂ hg)
  /-
    case h.e'_5.h
    α : Type u_1
    β : Type u_2
    σ : Type u_4
    inst✝² : Primcodable α
    inst✝¹ : Primcodable β
    inst✝ : Primcodable σ
    f : α → Option β
    g : α → β → σ
    hf : Computable f
    hg : Computable₂ g
    x✝ : α
    ⊢ Eq (Option.map (g x✝) (f x✝)) ((f x✝).bind fun b => Option.some (g x✝ b))
  -/
  apply Option.map_eq_bind
  /-
    🎉 no goals
  -/


theorem option_getD {f : α → Option β} {g : α → β} (hf : Computable f) (hg : Computable g) :
    Computable fun a => (f a).getD (g a) :=
  (Computable.option_casesOn hf hg (show Computable₂ fun _ b => b from Computable.snd)).of_eq
                /-
                  α : Type u_1
                  β : Type u_2
                  inst✝¹ : Primcodable α
                  inst✝ : Primcodable β
                  f : α → Option β
                  g : α → β
                  hf : Computable f
                  hg : Computable g
                  a : α
                  ⊢ Eq (Option.casesOn (f a) (g a) fun b => b) ((f a).getD (g a))
                -/
                              /-
                                🎉 no goals
                              -/
    fun a => by cases f a <;> rfl
                              /-
                                🎉 no goals
                              -/


theorem subtype_mk {f : α → β} {p : β → Prop} [DecidablePred p] {h : ∀ a, p (f a)}
    (hp : PrimrecPred p) (hf : Computable f) :
    @Computable _ _ _ (Primcodable.subtype hp) fun a => (⟨f a, h a⟩ : Subtype p) :=
  hf


theorem sum_casesOn {f : α → β ⊕ γ} {g : α → β → σ} {h : α → γ → σ} (hf : Computable f)
    (hg : Computable₂ g) (hh : Computable₂ h) :
    @Computable _ σ _ _ fun a => Sum.casesOn (f a) (g a) (h a) :=
  option_some_iff.1 <|
    (cond (nat_bodd.comp <| encode_iff.2 hf)
          (option_map (Computable.decode.comp <| nat_div2.comp <| encode_iff.2 hf) hh)
          (option_map (Computable.decode.comp <| nat_div2.comp <| encode_iff.2 hf) hg)).of_eq
      fun a => by
        /-
          α : Type u_1
          β : Type u_2
          γ : Type u_3
          σ : Type u_4
          inst✝³ : Primcodable α
          inst✝² : Primcodable β
          inst✝¹ : Primcodable γ
          inst✝ : Primcodable σ
          f : α → Sum β γ
          g : α → β → σ
          h : α → γ → σ
          hf : Computable f
          hg : Computable₂ g
          hh : Computable₂ h
          a : α
          ⊢ Eq (_root_.cond (Encodable.encode (f a)).bodd (Option.map (h a) (Encodable.d …
        -/
                                /-
                                  🎉 no goals
                                -/
        cases' f a with b c <;> simp [Nat.div2_val]
                                /-
                                  🎉 no goals
                                -/


theorem nat_strong_rec (f : α → ℕ → σ) {g : α → List σ → Option σ} (hg : Computable₂ g)
    (H : ∀ a n, g a ((List.range n).map (f a)) = Option.some (f a n)) : Computable₂ f :=
  suffices Computable₂ fun a n => (List.range n).map (f a) from
    option_some_iff.1 <|
      (list_get?.comp (this.comp fst (succ.comp snd)) snd).to₂.of_eq fun a => by
        /-
          α : Type u_1
          σ : Type u_4
          inst✝¹ : Primcodable α
          inst✝ : Primcodable σ
          f : α → Nat → σ
          g : α → List σ → Option σ
          hg : Computable₂ g
          H : ∀ (a : α) (n : Nat), Eq (g a (List.map (f a) (List.range n))) (Option.some …
          this : Computable₂ fun a n => List.map (f a) (List.range n)
          a : Prod α Nat
          ⊢ Eq ((fun a b => (List.map (f { fst := a, snd := b }.1) (List.range { fst :=  …
        -/
        simp [List.getElem?_range (Nat.lt_succ_self a.2)]
        /-
          🎉 no goals
        -/
  option_some_iff.1 <|
    (nat_rec snd (const (Option.some []))
          (to₂ <|
            option_bind (snd.comp snd) <|
              to₂ <|
                option_map (hg.comp (fst.comp <| fst.comp fst) snd)
      /-
        α : Type u_1
        σ : Type u_4
        inst✝¹ : Primcodable α
        inst✝ : Primcodable σ
        f : α → Nat → σ
        g : α → List σ → Option σ
        hg : Computable₂ g
        H : ∀ (a : α) (n : Nat), Eq (g a (List.map (f a) (List.range n))) (Option.some …
        a : Prod α Nat
        ⊢ Eq (Nat.rec (Option.some List.nil) (fun y IH => { fst := a, snd := { fst :=  …
      -/
                  (to₂ <| list_concat.comp (snd.comp fst) snd))).of_eq
                                  /-
                                    🎉 no goals
                                  -/
      /-
        case succ
        α : Type u_1
        σ : Type u_4
        inst✝¹ : Primcodable α
        inst✝ : Primcodable σ
        f : α → Nat → σ
        g : α → List σ → Option σ
        hg : Computable₂ g
        H : ∀ (a : α) (n : Nat), Eq (g a (List.map (f a) (List.range n))) (Option.some …
        a : Prod α Nat
        n : Nat
        IH : Eq (Nat.rec (Option.some List.nil) (fun y IH => { fst := a, snd := { fst  …
        ⊢ Eq (Nat.rec (Option.some List.nil) (fun y IH => { fst := a, snd := { fst :=  …
      -/
      fun a => by
      /-
        🎉 no goals
      -/
      induction' a.2 with n IH; · rfl
      simp [IH, H, List.range_succ]


theorem list_ofFn :
    ∀ {n} {f : Fin n → α → σ},
      (∀ i, Computable (f i)) → Computable fun a => List.ofFn fun i => f i a
  | 0, _, _ => by
    /-
      α : Type u_1
      σ : Type u_4
      inst✝¹ : Primcodable α
      inst✝ : Primcodable σ
      x✝¹ : Fin 0 → α → σ
      x✝ : ∀ (i : Fin 0), Computable (x✝¹ i)
      ⊢ Computable fun a => List.ofFn fun i => x✝¹ i a
    -/
    simp only [List.ofFn_zero]
    /-
      α : Type u_1
      σ : Type u_4
      inst✝¹ : Primcodable α
      inst✝ : Primcodable σ
      x✝¹ : Fin 0 → α → σ
      x✝ : ∀ (i : Fin 0), Computable (x✝¹ i)
      ⊢ Computable fun a => List.nil
    -/
    exact const []
    /-
      🎉 no goals
    -/
  | n + 1, f, hf => by
    /-
      α : Type u_1
      σ : Type u_4
      inst✝¹ : Primcodable α
      inst✝ : Primcodable σ
      n : Nat
      f : Fin (HAdd.hAdd n 1) → α → σ
      hf : ∀ (i : Fin (HAdd.hAdd n 1)), Computable (f i)
      ⊢ Computable fun a => List.ofFn fun i => f i a
    -/
    simp only [List.ofFn_succ]
    /-
      α : Type u_1
      σ : Type u_4
      inst✝¹ : Primcodable α
      inst✝ : Primcodable σ
      n : Nat
      f : Fin (HAdd.hAdd n 1) → α → σ
      hf : ∀ (i : Fin (HAdd.hAdd n 1)), Computable (f i)
      ⊢ Computable fun a => List.cons (f 0 a) (List.ofFn fun i => f i.succ a)
    -/
    exact list_cons.comp (hf 0) (list_ofFn fun i => hf i.succ)
    /-
      🎉 no goals
    -/


theorem vector_ofFn {n} {f : Fin n → α → σ} (hf : ∀ i, Computable (f i)) :
    Computable fun a => List.Vector.ofFn fun i => f i a :=
                                              /-
                                                α : Type u_1
                                                σ : Type u_4
                                                inst✝¹ : Primcodable α
                                                inst✝ : Primcodable σ
                                                n : Nat
                                                f : Fin n → α → σ
                                                hf : ∀ (i : Fin n), Computable (f i)
                                                a : α
                                                ⊢ Eq (List.Vector.mOfFn fun i => ↑(f i) a) ((↑fun a => List.Vector.ofFn fun i  …
                                              -/
  (Partrec.vector_mOfFn hf).of_eq fun a => by simp
                                              /-
                                                🎉 no goals
                                              -/


theorem option_some_iff {f : α →. σ} : (Partrec fun a => (f a).map Option.some) ↔ Partrec f :=
  ⟨fun h => (Nat.Partrec.ppred.comp h).of_eq fun n => by
      -- Porting note: needed to help with applying bind_some_eq_map because `Function.comp` got
      -- less reducible.
      simp [Part.bind_assoc, ← Function.comp_apply (f := Part.some) (g := encode), bind_some_eq_map,
        -Function.comp_apply],
    fun hf => hf.map (option_some.comp snd).to₂⟩


theorem option_casesOn_right {o : α → Option β} {f : α → σ} {g : α → β →. σ} (ho : Computable o)
    (hf : Computable f) (hg : Partrec₂ g) :
    @Partrec _ σ _ _ fun a => Option.casesOn (o a) (Part.some (f a)) (g a) :=
  have :
    Partrec fun a : α =>
      Nat.casesOn (encode (o a)) (Part.some (f a)) (fun n => Part.bind (decode (α := β) n) (g a)) :=
    nat_casesOn_right (h := fun a n ↦ Part.bind (ofOption (decode n)) fun b ↦ g a b)
      (encode_iff.2 ho) hf.partrec <|
        ((@Computable.decode β _).comp snd).ofOption.bind (hg.comp (fst.comp fst) snd).to₂
                         /-
                           α : Type u_1
                           β : Type u_2
                           σ : Type u_4
                           inst✝² : Primcodable α
                           inst✝¹ : Primcodable β
                           inst✝ : Primcodable σ
                           o : α → Option β
                           f : α → σ
                           g : α → PFun β σ
                           ho : Computable o
                           hf : Computable f
                           hg : Partrec₂ g
                           this : Partrec fun a => Nat.casesOn (Encodable.encode (o a)) (Part.some (f a)) …
                           a : α
                           ⊢ Eq (Nat.casesOn (Encodable.encode (o a)) (Part.some (f a)) fun n => (↑(Encod …
                         -/
                                               /-
                                                 🎉 no goals
                                               -/
  this.of_eq fun a => by cases' o a with b <;> simp [encodek]
                                               /-
                                                 🎉 no goals
                                               -/


theorem sum_casesOn_right {f : α → β ⊕ γ} {g : α → β → σ} {h : α → γ →. σ} (hf : Computable f)
    (hg : Computable₂ g) (hh : Partrec₂ h) :
    @Partrec _ σ _ _ fun a => Sum.casesOn (f a) (fun b => Part.some (g a b)) (h a) :=
  have :
    Partrec fun a =>
      (Option.casesOn (Sum.casesOn (f a) (fun _ => Option.none) Option.some : Option γ)
          (some (Sum.casesOn (f a) (fun b => some (g a b)) fun _ => Option.none)) fun c =>
          (h a c).map Option.some :
        Part (Option σ)) :=
    option_casesOn_right (g := fun a n => Part.map Option.some (h a n))
      (sum_casesOn hf (const Option.none).to₂ (option_some.comp snd).to₂)
      (sum_casesOn (g := fun a n => Option.some (g a n)) hf (option_some.comp hg)
        (const Option.none).to₂)
      (option_some_iff.2 hh)
                                              /-
                                                α : Type u_1
                                                β : Type u_2
                                                γ : Type u_3
                                                σ : Type u_4
                                                inst✝³ : Primcodable α
                                                inst✝² : Primcodable β
                                                inst✝¹ : Primcodable γ
                                                inst✝ : Primcodable σ
                                                f : α → Sum β γ
                                                g : α → β → σ
                                                h : α → PFun γ σ
                                                hf : Computable f
                                                hg : Computable₂ g
                                                hh : Partrec₂ h
                                                this : Partrec fun a => Option.casesOn (Sum.casesOn (f a) (fun x => Option.non …
                                                a : α
                                                ⊢ Eq (Option.casesOn (Sum.casesOn (f a) (fun x => Option.none) Option.some) (P …
                                              -/
                                                            /-
                                                              🎉 no goals
                                                            -/
  option_some_iff.1 <| this.of_eq fun a => by cases f a <;> simp
                                                            /-
                                                              🎉 no goals
                                                            -/


theorem sum_casesOn_left {f : α → β ⊕ γ} {g : α → β →. σ} {h : α → γ → σ} (hf : Computable f)
    (hg : Partrec₂ g) (hh : Computable₂ h) :
    @Partrec _ σ _ _ fun a => Sum.casesOn (f a) (g a) fun c => Part.some (h a c) :=
  (sum_casesOn_right (sum_casesOn hf (sum_inr.comp snd).to₂ (sum_inl.comp snd).to₂) hh hg).of_eq
                /-
                  α : Type u_1
                  β : Type u_2
                  γ : Type u_3
                  σ : Type u_4
                  inst✝³ : Primcodable α
                  inst✝² : Primcodable β
                  inst✝¹ : Primcodable γ
                  inst✝ : Primcodable σ
                  f : α → Sum β γ
                  g : α → PFun β σ
                  h : α → γ → σ
                  hf : Computable f
                  hg : Partrec₂ g
                  hh : Computable₂ h
                  a : α
                  ⊢ Eq (Sum.casesOn (Sum.casesOn (f a) (fun b => Sum.inr { fst := a, snd := b }. …
                -/
                              /-
                                🎉 no goals
                              -/
    fun a => by cases f a <;> simp
                              /-
                                🎉 no goals
                              -/


theorem fix_aux {α σ} (f : α →. σ ⊕ α) (a : α) (b : σ) :
    let F : α → ℕ →. σ ⊕ α := fun a n =>
      n.rec (some (Sum.inr a)) fun _ IH => IH.bind fun s => Sum.casesOn s (fun _ => Part.some s) f
    (∃ n : ℕ,
        ((∃ b' : σ, Sum.inl b' ∈ F a n) ∧ ∀ {m : ℕ}, m < n → ∃ b : α, Sum.inr b ∈ F a m) ∧
          Sum.inl b ∈ F a n) ↔
      b ∈ PFun.fix f a := by
  /-
    α : Type u_5
    σ : Type u_6
    f : PFun α (Sum σ α)
    a : α
    b : σ
    ⊢ let F := fun a n => Nat.rec (Part.some (Sum.inr a)) (fun x IH => IH.bind fun …
      Iff (Exists fun n => And (And (Exists fun b' => Membership.mem (F a n) (Sum. …
  -/
  intro F; refine ⟨fun h => ?_, fun h => ?_⟩
    /-
      case refine_1
      α : Type u_5
      σ : Type u_6
      f : PFun α (Sum σ α)
      a : α
      b : σ
      F : α → PFun Nat (Sum σ α) := fun a n => Nat.rec (Part.some (Sum.inr a)) (fun  …
      h : Exists fun n => And (And (Exists fun b' => Membership.mem (F a n) (Sum.inl …
      ⊢ Membership.mem (f.fix a) b
    -/
  · rcases h with ⟨n, ⟨_x, h₁⟩, h₂⟩
    have : ∀ m a', Sum.inr a' ∈ F a m → b ∈ PFun.fix f a' → b ∈ PFun.fix f a := by
      intro m a' am ba
      induction' m with m IH generalizing a' <;> simp [F] at am
      · rwa [← am]
      rcases am with ⟨a₂, am₂, fa₂⟩
      exact IH _ am₂ (PFun.mem_fix_iff.2 (Or.inr ⟨_, fa₂, ba⟩))
    /-
      case refine_1.intro.intro.intro
      α : Type u_5
      σ : Type u_6
      f : PFun α (Sum σ α)
      a : α
      b : σ
      F : α → PFun Nat (Sum σ α) := fun a n => Nat.rec (Part.some (Sum.inr a)) (fun  …
      n : Nat
      h₂ : Membership.mem (F a n) (Sum.inl b)
      _x : Exists fun b' => Membership.mem (F a n) (Sum.inl b')
      h₁ : ∀ {m : Nat}, LT.lt m n → Exists fun b => Membership.mem (F a m) (Sum.inr b)
      this : ∀ (m : Nat) (a' : α), Membership.mem (F a m) (Sum.inr a') → Membership. …
      ⊢ Membership.mem (f.fix a) b
    -/
                /-
                  🎉 no goals
                -/
    cases n <;> simp [F] at h₂
    /-
      case refine_1.intro.intro.intro.succ
      α : Type u_5
      σ : Type u_6
      f : PFun α (Sum σ α)
      a : α
      b : σ
      F : α → PFun Nat (Sum σ α) := fun a n => Nat.rec (Part.some (Sum.inr a)) (fun  …
      this : ∀ (m : Nat) (a' : α), Membership.mem (F a m) (Sum.inr a') → Membership. …
      n✝ : Nat
      _x : Exists fun b' => Membership.mem (F a (HAdd.hAdd n✝ 1)) (Sum.inl b')
      h₁ : ∀ {m : Nat}, LT.lt m (HAdd.hAdd n✝ 1) → Exists fun b => Membership.mem (F …
      h₂ : Or (Membership.mem (Nat.rec (Part.some (Sum.inr a)) (fun x IH => IH.bind  …
      ⊢ Membership.mem (f.fix a) b
    -/
    rcases h₂ with (h₂ | ⟨a', am', fa'⟩)
      /-
        case refine_1.intro.intro.intro.succ.inl
        α : Type u_5
        σ : Type u_6
        f : PFun α (Sum σ α)
        a : α
        b : σ
        F : α → PFun Nat (Sum σ α) := fun a n => Nat.rec (Part.some (Sum.inr a)) (fun  …
        this : ∀ (m : Nat) (a' : α), Membership.mem (F a m) (Sum.inr a') → Membership. …
        n✝ : Nat
        _x : Exists fun b' => Membership.mem (F a (HAdd.hAdd n✝ 1)) (Sum.inl b')
        h₁ : ∀ {m : Nat}, LT.lt m (HAdd.hAdd n✝ 1) → Exists fun b => Membership.mem (F …
        h₂ : Membership.mem (Nat.rec (Part.some (Sum.inr a)) (fun x IH => IH.bind fun  …
        ⊢ Membership.mem (f.fix a) b
      -/
    · cases' h₁ (Nat.lt_succ_self _) with a' h
      /-
        case refine_1.intro.intro.intro.succ.inl.intro
        α : Type u_5
        σ : Type u_6
        f : PFun α (Sum σ α)
        a : α
        b : σ
        F : α → PFun Nat (Sum σ α) := fun a n => Nat.rec (Part.some (Sum.inr a)) (fun  …
        this : ∀ (m : Nat) (a' : α), Membership.mem (F a m) (Sum.inr a') → Membership. …
        n✝ : Nat
        _x : Exists fun b' => Membership.mem (F a (HAdd.hAdd n✝ 1)) (Sum.inl b')
        h₁ : ∀ {m : Nat}, LT.lt m (HAdd.hAdd n✝ 1) → Exists fun b => Membership.mem (F …
        h₂ : Membership.mem (Nat.rec (Part.some (Sum.inr a)) (fun x IH => IH.bind fun  …
        a' : α
        h : Membership.mem (F a n✝) (Sum.inr a')
        ⊢ Membership.mem (f.fix a) b
      -/
      injection mem_unique h h₂
      /-
        🎉 no goals
      -/
      /-
        case refine_1.intro.intro.intro.succ.inr.intro.intro
        α : Type u_5
        σ : Type u_6
        f : PFun α (Sum σ α)
        a : α
        b : σ
        F : α → PFun Nat (Sum σ α) := fun a n => Nat.rec (Part.some (Sum.inr a)) (fun  …
        this : ∀ (m : Nat) (a' : α), Membership.mem (F a m) (Sum.inr a') → Membership. …
        n✝ : Nat
        _x : Exists fun b' => Membership.mem (F a (HAdd.hAdd n✝ 1)) (Sum.inl b')
        h₁ : ∀ {m : Nat}, LT.lt m (HAdd.hAdd n✝ 1) → Exists fun b => Membership.mem (F …
        a' : α
        am' : Membership.mem (Nat.rec (Part.some (Sum.inr a)) (fun x IH => IH.bind fun …
        fa' : Membership.mem (f a') (Sum.inl b)
        ⊢ Membership.mem (f.fix a) b
      -/
    · exact this _ _ am' (PFun.mem_fix_iff.2 (Or.inl fa'))
      /-
        🎉 no goals
      -/
  · suffices ∀ a', b ∈ PFun.fix f a' → ∀ k, Sum.inr a' ∈ F a k →
        ∃ n, Sum.inl b ∈ F a n ∧ ∀ m < n, k ≤ m → ∃ a₂, Sum.inr a₂ ∈ F a m by
      rcases this _ h 0 (by simp [F]) with ⟨n, hn₁, hn₂⟩
      exact ⟨_, ⟨⟨_, hn₁⟩, fun {m} mn => hn₂ m mn (Nat.zero_le _)⟩, hn₁⟩
    /-
      case refine_2
      α : Type u_5
      σ : Type u_6
      f : PFun α (Sum σ α)
      a : α
      b : σ
      F : α → PFun Nat (Sum σ α) := fun a n => Nat.rec (Part.some (Sum.inr a)) (fun  …
      h : Membership.mem (f.fix a) b
      ⊢ ∀ (a' : α), Membership.mem (f.fix a') b → ∀ (k : Nat), Membership.mem (F a k …
    -/
    intro a₁ h₁
    /-
      case refine_2
      α : Type u_5
      σ : Type u_6
      f : PFun α (Sum σ α)
      a : α
      b : σ
      F : α → PFun Nat (Sum σ α) := fun a n => Nat.rec (Part.some (Sum.inr a)) (fun  …
      h : Membership.mem (f.fix a) b
      a₁ : α
      h₁ : Membership.mem (f.fix a₁) b
      ⊢ ∀ (k : Nat), Membership.mem (F a k) (Sum.inr a₁) → Exists fun n => And (Memb …
    -/
    apply @PFun.fixInduction _ _ _ _ _ _ h₁
    /-
      case refine_2
      α : Type u_5
      σ : Type u_6
      f : PFun α (Sum σ α)
      a : α
      b : σ
      F : α → PFun Nat (Sum σ α) := fun a n => Nat.rec (Part.some (Sum.inr a)) (fun  …
      h : Membership.mem (f.fix a) b
      a₁ : α
      h₁ : Membership.mem (f.fix a₁) b
      ⊢ ∀ (a' : α), Membership.mem (f.fix a') b → (∀ (a'' : α), Membership.mem (f a' …
    -/
    intro a₂ h₂ IH k hk
    /-
      case refine_2
      α : Type u_5
      σ : Type u_6
      f : PFun α (Sum σ α)
      a : α
      b : σ
      F : α → PFun Nat (Sum σ α) := fun a n => Nat.rec (Part.some (Sum.inr a)) (fun  …
      h : Membership.mem (f.fix a) b
      a₁ : α
      h₁ : Membership.mem (f.fix a₁) b
      a₂ : α
      h₂ : Membership.mem (f.fix a₂) b
      IH : ∀ (a'' : α), Membership.mem (f a₂) (Sum.inr a'') → ∀ (k : Nat), Membershi …
      k : Nat
      hk : Membership.mem (F a k) (Sum.inr a₂)
      ⊢ Exists fun n => And (Membership.mem (F a n) (Sum.inl b)) (∀ (m : Nat), LT.lt …
    -/
    rcases PFun.mem_fix_iff.1 h₂ with (h₂ | ⟨a₃, am₃, _⟩)
      /-
        case refine_2.inl
        α : Type u_5
        σ : Type u_6
        f : PFun α (Sum σ α)
        a : α
        b : σ
        F : α → PFun Nat (Sum σ α) := fun a n => Nat.rec (Part.some (Sum.inr a)) (fun  …
        h : Membership.mem (f.fix a) b
        a₁ : α
        h₁ : Membership.mem (f.fix a₁) b
        a₂ : α
        h₂✝ : Membership.mem (f.fix a₂) b
        IH : ∀ (a'' : α), Membership.mem (f a₂) (Sum.inr a'') → ∀ (k : Nat), Membershi …
        k : Nat
        hk : Membership.mem (F a k) (Sum.inr a₂)
        h₂ : Membership.mem (f a₂) (Sum.inl b)
        ⊢ Exists fun n => And (Membership.mem (F a n) (Sum.inl b)) (∀ (m : Nat), LT.lt …
      -/
    · refine ⟨k.succ, ?_, fun m mk km => ⟨a₂, ?_⟩⟩
        /-
          case refine_2.inl.refine_1
          α : Type u_5
          σ : Type u_6
          f : PFun α (Sum σ α)
          a : α
          b : σ
          F : α → PFun Nat (Sum σ α) := fun a n => Nat.rec (Part.some (Sum.inr a)) (fun  …
          h : Membership.mem (f.fix a) b
          a₁ : α
          h₁ : Membership.mem (f.fix a₁) b
          a₂ : α
          h₂✝ : Membership.mem (f.fix a₂) b
          IH : ∀ (a'' : α), Membership.mem (f a₂) (Sum.inr a'') → ∀ (k : Nat), Membershi …
          k : Nat
          hk : Membership.mem (F a k) (Sum.inr a₂)
          h₂ : Membership.mem (f a₂) (Sum.inl b)
          ⊢ Membership.mem (F a k.succ) (Sum.inl b)
        -/
      · simpa [F] using Or.inr ⟨_, hk, h₂⟩
        /-
          🎉 no goals
        -/
        /-
          case refine_2.inl.refine_2
          α : Type u_5
          σ : Type u_6
          f : PFun α (Sum σ α)
          a : α
          b : σ
          F : α → PFun Nat (Sum σ α) := fun a n => Nat.rec (Part.some (Sum.inr a)) (fun  …
          h : Membership.mem (f.fix a) b
          a₁ : α
          h₁ : Membership.mem (f.fix a₁) b
          a₂ : α
          h₂✝ : Membership.mem (f.fix a₂) b
          IH : ∀ (a'' : α), Membership.mem (f a₂) (Sum.inr a'') → ∀ (k : Nat), Membershi …
          k : Nat
          hk : Membership.mem (F a k) (Sum.inr a₂)
          h₂ : Membership.mem (f a₂) (Sum.inl b)
          m : Nat
          mk : LT.lt m k.succ
          km : LE.le k m
          ⊢ Membership.mem (F a m) (Sum.inr a₂)
        -/
      · rwa [le_antisymm (Nat.le_of_lt_succ mk) km]
        /-
          🎉 no goals
        -/
      /-
        case refine_2.inr.intro.intro
        α : Type u_5
        σ : Type u_6
        f : PFun α (Sum σ α)
        a : α
        b : σ
        F : α → PFun Nat (Sum σ α) := fun a n => Nat.rec (Part.some (Sum.inr a)) (fun  …
        h : Membership.mem (f.fix a) b
        a₁ : α
        h₁ : Membership.mem (f.fix a₁) b
        a₂ : α
        h₂ : Membership.mem (f.fix a₂) b
        IH : ∀ (a'' : α), Membership.mem (f a₂) (Sum.inr a'') → ∀ (k : Nat), Membershi …
        k : Nat
        hk : Membership.mem (F a k) (Sum.inr a₂)
        a₃ : α
        am₃ : Membership.mem (f a₂) (Sum.inr a₃)
        right✝ : Membership.mem (f.fix a₃) b
        ⊢ Exists fun n => And (Membership.mem (F a n) (Sum.inl b)) (∀ (m : Nat), LT.lt …
      -/
    · rcases IH _ am₃ k.succ (by simpa [F] using ⟨_, hk, am₃⟩) with ⟨n, hn₁, hn₂⟩
      /-
        case refine_2.inr.intro.intro.intro.intro
        α : Type u_5
        σ : Type u_6
        f : PFun α (Sum σ α)
        a : α
        b : σ
        F : α → PFun Nat (Sum σ α) := fun a n => Nat.rec (Part.some (Sum.inr a)) (fun  …
        h : Membership.mem (f.fix a) b
        a₁ : α
        h₁ : Membership.mem (f.fix a₁) b
        a₂ : α
        h₂ : Membership.mem (f.fix a₂) b
        IH : ∀ (a'' : α), Membership.mem (f a₂) (Sum.inr a'') → ∀ (k : Nat), Membershi …
        k : Nat
        hk : Membership.mem (F a k) (Sum.inr a₂)
        a₃ : α
        am₃ : Membership.mem (f a₂) (Sum.inr a₃)
        right✝ : Membership.mem (f.fix a₃) b
        n : Nat
        hn₁ : Membership.mem (F a n) (Sum.inl b)
        hn₂ : ∀ (m : Nat), LT.lt m n → LE.le k.succ m → Exists fun a₂ => Membership.me …
        ⊢ Exists fun n => And (Membership.mem (F a n) (Sum.inl b)) (∀ (m : Nat), LT.lt …
      -/
      refine ⟨n, hn₁, fun m mn km => ?_⟩
      /-
        case refine_2.inr.intro.intro.intro.intro
        α : Type u_5
        σ : Type u_6
        f : PFun α (Sum σ α)
        a : α
        b : σ
        F : α → PFun Nat (Sum σ α) := fun a n => Nat.rec (Part.some (Sum.inr a)) (fun  …
        h : Membership.mem (f.fix a) b
        a₁ : α
        h₁ : Membership.mem (f.fix a₁) b
        a₂ : α
        h₂ : Membership.mem (f.fix a₂) b
        IH : ∀ (a'' : α), Membership.mem (f a₂) (Sum.inr a'') → ∀ (k : Nat), Membershi …
        k : Nat
        hk : Membership.mem (F a k) (Sum.inr a₂)
        a₃ : α
        am₃ : Membership.mem (f a₂) (Sum.inr a₃)
        right✝ : Membership.mem (f.fix a₃) b
        n : Nat
        hn₁ : Membership.mem (F a n) (Sum.inl b)
        hn₂ : ∀ (m : Nat), LT.lt m n → LE.le k.succ m → Exists fun a₂ => Membership.me …
        m : Nat
        mn : LT.lt m n
        km : LE.le k m
        ⊢ Exists fun a₂ => Membership.mem (F a m) (Sum.inr a₂)
      -/
      cases' km.lt_or_eq_dec with km km
        /-
          case refine_2.inr.intro.intro.intro.intro.inl
          α : Type u_5
          σ : Type u_6
          f : PFun α (Sum σ α)
          a : α
          b : σ
          F : α → PFun Nat (Sum σ α) := fun a n => Nat.rec (Part.some (Sum.inr a)) (fun  …
          h : Membership.mem (f.fix a) b
          a₁ : α
          h₁ : Membership.mem (f.fix a₁) b
          a₂ : α
          h₂ : Membership.mem (f.fix a₂) b
          IH : ∀ (a'' : α), Membership.mem (f a₂) (Sum.inr a'') → ∀ (k : Nat), Membershi …
          k : Nat
          hk : Membership.mem (F a k) (Sum.inr a₂)
          a₃ : α
          am₃ : Membership.mem (f a₂) (Sum.inr a₃)
          right✝ : Membership.mem (f.fix a₃) b
          n : Nat
          hn₁ : Membership.mem (F a n) (Sum.inl b)
          hn₂ : ∀ (m : Nat), LT.lt m n → LE.le k.succ m → Exists fun a₂ => Membership.me …
          m : Nat
          mn : LT.lt m n
          km✝ : LE.le k m
          km : LT.lt k m
          ⊢ Exists fun a₂ => Membership.mem (F a m) (Sum.inr a₂)
        -/
      · exact hn₂ _ mn km
        /-
          🎉 no goals
        -/
        /-
          case refine_2.inr.intro.intro.intro.intro.inr
          α : Type u_5
          σ : Type u_6
          f : PFun α (Sum σ α)
          a : α
          b : σ
          F : α → PFun Nat (Sum σ α) := fun a n => Nat.rec (Part.some (Sum.inr a)) (fun  …
          h : Membership.mem (f.fix a) b
          a₁ : α
          h₁ : Membership.mem (f.fix a₁) b
          a₂ : α
          h₂ : Membership.mem (f.fix a₂) b
          IH : ∀ (a'' : α), Membership.mem (f a₂) (Sum.inr a'') → ∀ (k : Nat), Membershi …
          k : Nat
          hk : Membership.mem (F a k) (Sum.inr a₂)
          a₃ : α
          am₃ : Membership.mem (f a₂) (Sum.inr a₃)
          right✝ : Membership.mem (f.fix a₃) b
          n : Nat
          hn₁ : Membership.mem (F a n) (Sum.inl b)
          hn₂ : ∀ (m : Nat), LT.lt m n → LE.le k.succ m → Exists fun a₂ => Membership.me …
          m : Nat
          mn : LT.lt m n
          km✝ : LE.le k m
          km : Eq k m
          ⊢ Exists fun a₂ => Membership.mem (F a m) (Sum.inr a₂)
        -/
      · exact km ▸ ⟨_, hk⟩
        /-
          🎉 no goals
        -/


theorem fix {f : α →. σ ⊕ α} (hf : Partrec f) : Partrec (PFun.fix f) := by
  let F : α → ℕ →. σ ⊕ α := fun a n =>
    n.rec (some (Sum.inr a)) fun _ IH => IH.bind fun s => Sum.casesOn s (fun _ => Part.some s) f
  have hF : Partrec₂ F :=
    Partrec.nat_rec snd (sum_inr.comp fst).partrec
      (sum_casesOn_right (snd.comp snd) (snd.comp <| snd.comp fst).to₂ (hf.comp snd).to₂).to₂
  /-
    α : Type u_1
    σ : Type u_4
    inst✝¹ : Primcodable α
    inst✝ : Primcodable σ
    f : PFun α (Sum σ α)
    hf : Partrec f
    F : α → PFun Nat (Sum σ α) := fun a n => Nat.rec (Part.some (Sum.inr a)) (fun  …
    hF : Partrec₂ F
    ⊢ Partrec f.fix
  -/
  let p a n := @Part.map _ Bool (fun s => Sum.casesOn s (fun _ => true) fun _ => false) (F a n)
  have hp : Partrec₂ p :=
    hF.map ((sum_casesOn Computable.id (const true).to₂ (const false).to₂).comp snd).to₂
  exact (hp.rfind.bind (hF.bind (sum_casesOn_right snd snd.to₂ none.to₂).to₂).to₂).of_eq fun a =>
    ext fun b => by simpa [p] using fix_aux f _ _


