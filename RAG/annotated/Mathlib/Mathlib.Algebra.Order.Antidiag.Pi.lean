/-- `finAntidiagonal d n` is the type of `d`-tuples with sum `n`.

TODO: deduplicate with the less general `Finset.Nat.antidiagonalTuple`. -/
def finAntidiagonal (d : ℕ) (n : μ) : Finset (Fin d → μ) :=
  aux d n
where
  /-- Auxiliary construction for `finAntidiagonal` that bundles a proof of lawfulness
  (`mem_finAntidiagonal`), as this is needed to invoke `disjiUnion`. Using `Finset.disjiUnion` makes
  this computationally much more efficient than using `Finset.biUnion`. -/
  aux (d : ℕ) (n : μ) : {s : Finset (Fin d → μ) // ∀ f, f ∈ s ↔ ∑ i, f i = n} :=
    match d with
    | 0 =>
      if h : n = 0 then
                 /-
                   ι : Type u_1
                   μ : Type u_2
                   μ' : Type u_3
                   inst✝³ : DecidableEq ι
                   inst✝² : AddCommMonoid μ
                   inst✝¹ : Finset.HasAntidiagonal μ
                   inst✝ : DecidableEq μ
                   n✝¹ : μ
                   d✝ : Nat
                   n✝ : μ
                   d : Nat
                   n : μ
                   h : Eq n 0
                   ⊢ ∀ (f : Fin 0 → μ), Iff (Membership.mem (Singleton.singleton 0) f) (Eq (Finse …
                 -/
        ⟨{0}, by simp [h, Subsingleton.elim _ ![]]⟩
                 /-
                   🎉 no goals
                 -/
      else
               /-
                 ι : Type u_1
                 μ : Type u_2
                 μ' : Type u_3
                 inst✝³ : DecidableEq ι
                 inst✝² : AddCommMonoid μ
                 inst✝¹ : Finset.HasAntidiagonal μ
                 inst✝ : DecidableEq μ
                 n✝¹ : μ
                 d✝ : Nat
                 n✝ : μ
                 d : Nat
                 n : μ
                 h : Not (Eq n 0)
                 ⊢ ∀ (f : Fin 0 → μ), Iff (Membership.mem EmptyCollection.emptyCollection f) (E …
               -/
        ⟨∅, by simp [Ne.symm h]⟩
               /-
                 🎉 no goals
               -/
    | d + 1 =>
      { val := (antidiagonal n).disjiUnion
          (fun ab => (aux d ab.2).1.map {
              toFun := Fin.cons (ab.1)
              inj' := Fin.cons_right_injective _ })
          (fun i _hi j _hj hij => Finset.disjoint_left.2 fun t hti htj => hij <| by
            /-
              ι : Type u_1
              μ : Type u_2
              μ' : Type u_3
              inst✝³ : DecidableEq ι
              inst✝² : AddCommMonoid μ
              inst✝¹ : Finset.HasAntidiagonal μ
              inst✝ : DecidableEq μ
              n✝¹ : μ
              d✝¹ : Nat
              n✝ : μ
              d✝ : Nat
              n : μ
              d : Nat
              i : Prod μ μ
              _hi : Membership.mem (↑(Finset.HasAntidiagonal.antidiagonal n)) i
              j : Prod μ μ
              _hj : Membership.mem (↑(Finset.HasAntidiagonal.antidiagonal n)) j
              hij : Ne i j
              t : Fin (HAdd.hAdd d 1) → μ
              hti : Membership.mem ((fun ab => Finset.map { toFun := Fin.cons ab.1, inj' :=  …
              htj : Membership.mem ((fun ab => Finset.map { toFun := Fin.cons ab.1, inj' :=  …
              ⊢ Eq i j
            -/
            simp_rw [Finset.mem_map, Embedding.coeFn_mk] at hti htj
            /-
              ι : Type u_1
              μ : Type u_2
              μ' : Type u_3
              inst✝³ : DecidableEq ι
              inst✝² : AddCommMonoid μ
              inst✝¹ : Finset.HasAntidiagonal μ
              inst✝ : DecidableEq μ
              n✝¹ : μ
              d✝¹ : Nat
              n✝ : μ
              d✝ : Nat
              n : μ
              d : Nat
              i : Prod μ μ
              _hi : Membership.mem (↑(Finset.HasAntidiagonal.antidiagonal n)) i
              j : Prod μ μ
              _hj : Membership.mem (↑(Finset.HasAntidiagonal.antidiagonal n)) j
              hij : Ne i j
              t : Fin (HAdd.hAdd d 1) → μ
              hti : Exists fun a => And (Membership.mem (↑(aux d i.2)) a) (Eq (Fin.cons i.1  …
              htj : Exists fun a => And (Membership.mem (↑(aux d j.2)) a) (Eq (Fin.cons j.1  …
              ⊢ Eq i j
            -/
            obtain ⟨ai, hai, hij'⟩ := hti
            /-
              case intro.intro
              ι : Type u_1
              μ : Type u_2
              μ' : Type u_3
              inst✝³ : DecidableEq ι
              inst✝² : AddCommMonoid μ
              inst✝¹ : Finset.HasAntidiagonal μ
              inst✝ : DecidableEq μ
              n✝¹ : μ
              d✝¹ : Nat
              n✝ : μ
              d✝ : Nat
              n : μ
              d : Nat
              i : Prod μ μ
              _hi : Membership.mem (↑(Finset.HasAntidiagonal.antidiagonal n)) i
              j : Prod μ μ
              _hj : Membership.mem (↑(Finset.HasAntidiagonal.antidiagonal n)) j
              hij : Ne i j
              t : Fin (HAdd.hAdd d 1) → μ
              htj : Exists fun a => And (Membership.mem (↑(aux d j.2)) a) (Eq (Fin.cons j.1  …
              ai : Fin d → μ
              hai : Membership.mem (↑(aux d i.2)) ai
              hij' : Eq (Fin.cons i.1 ai) t
              ⊢ Eq i j
            -/
            obtain ⟨aj, haj, rfl⟩ := htj
            /-
              case intro.intro.intro.intro
              ι : Type u_1
              μ : Type u_2
              μ' : Type u_3
              inst✝³ : DecidableEq ι
              inst✝² : AddCommMonoid μ
              inst✝¹ : Finset.HasAntidiagonal μ
              inst✝ : DecidableEq μ
              n✝¹ : μ
              d✝¹ : Nat
              n✝ : μ
              d✝ : Nat
              n : μ
              d : Nat
              i : Prod μ μ
              _hi : Membership.mem (↑(Finset.HasAntidiagonal.antidiagonal n)) i
              j : Prod μ μ
              _hj : Membership.mem (↑(Finset.HasAntidiagonal.antidiagonal n)) j
              hij : Ne i j
              ai : Fin d → μ
              hai : Membership.mem (↑(aux d i.2)) ai
              aj : Fin d → μ
              haj : Membership.mem (↑(aux d j.2)) aj
              hij' : Eq (Fin.cons i.1 ai) (Fin.cons j.1 aj)
              ⊢ Eq i j
            -/
            rw [Fin.cons_eq_cons] at hij'
            /-
              case intro.intro.intro.intro
              ι : Type u_1
              μ : Type u_2
              μ' : Type u_3
              inst✝³ : DecidableEq ι
              inst✝² : AddCommMonoid μ
              inst✝¹ : Finset.HasAntidiagonal μ
              inst✝ : DecidableEq μ
              n✝¹ : μ
              d✝¹ : Nat
              n✝ : μ
              d✝ : Nat
              n : μ
              d : Nat
              i : Prod μ μ
              _hi : Membership.mem (↑(Finset.HasAntidiagonal.antidiagonal n)) i
              j : Prod μ μ
              _hj : Membership.mem (↑(Finset.HasAntidiagonal.antidiagonal n)) j
              hij : Ne i j
              ai : Fin d → μ
              hai : Membership.mem (↑(aux d i.2)) ai
              aj : Fin d → μ
              haj : Membership.mem (↑(aux d j.2)) aj
              hij' : And (Eq i.1 j.1) (Eq ai aj)
              ⊢ Eq i j
            -/
            ext
              /-
                case intro.intro.intro.intro.fst
                ι : Type u_1
                μ : Type u_2
                μ' : Type u_3
                inst✝³ : DecidableEq ι
                inst✝² : AddCommMonoid μ
                inst✝¹ : Finset.HasAntidiagonal μ
                inst✝ : DecidableEq μ
                n✝¹ : μ
                d✝¹ : Nat
                n✝ : μ
                d✝ : Nat
                n : μ
                d : Nat
                i : Prod μ μ
                _hi : Membership.mem (↑(Finset.HasAntidiagonal.antidiagonal n)) i
                j : Prod μ μ
                _hj : Membership.mem (↑(Finset.HasAntidiagonal.antidiagonal n)) j
                hij : Ne i j
                ai : Fin d → μ
                hai : Membership.mem (↑(aux d i.2)) ai
                aj : Fin d → μ
                haj : Membership.mem (↑(aux d j.2)) aj
                hij' : And (Eq i.1 j.1) (Eq ai aj)
                ⊢ Eq i.1 j.1
              -/
            · exact hij'.1
              /-
                🎉 no goals
              -/
              /-
                case intro.intro.intro.intro.snd
                ι : Type u_1
                μ : Type u_2
                μ' : Type u_3
                inst✝³ : DecidableEq ι
                inst✝² : AddCommMonoid μ
                inst✝¹ : Finset.HasAntidiagonal μ
                inst✝ : DecidableEq μ
                n✝¹ : μ
                d✝¹ : Nat
                n✝ : μ
                d✝ : Nat
                n : μ
                d : Nat
                i : Prod μ μ
                _hi : Membership.mem (↑(Finset.HasAntidiagonal.antidiagonal n)) i
                j : Prod μ μ
                _hj : Membership.mem (↑(Finset.HasAntidiagonal.antidiagonal n)) j
                hij : Ne i j
                ai : Fin d → μ
                hai : Membership.mem (↑(aux d i.2)) ai
                aj : Fin d → μ
                haj : Membership.mem (↑(aux d j.2)) aj
                hij' : And (Eq i.1 j.1) (Eq ai aj)
                ⊢ Eq i.2 j.2
              -/
            · obtain ⟨-, rfl⟩ := hij'
              /-
                case intro.intro.intro.intro.snd.intro
                ι : Type u_1
                μ : Type u_2
                μ' : Type u_3
                inst✝³ : DecidableEq ι
                inst✝² : AddCommMonoid μ
                inst✝¹ : Finset.HasAntidiagonal μ
                inst✝ : DecidableEq μ
                n✝¹ : μ
                d✝¹ : Nat
                n✝ : μ
                d✝ : Nat
                n : μ
                d : Nat
                i : Prod μ μ
                _hi : Membership.mem (↑(Finset.HasAntidiagonal.antidiagonal n)) i
                j : Prod μ μ
                _hj : Membership.mem (↑(Finset.HasAntidiagonal.antidiagonal n)) j
                hij : Ne i j
                ai : Fin d → μ
                hai : Membership.mem (↑(aux d i.2)) ai
                haj : Membership.mem (↑(aux d j.2)) ai
                ⊢ Eq i.2 j.2
              -/
              rw [← (aux d i.2).prop ai |>.mp hai, ← (aux d j.2).prop ai |>.mp haj])
              /-
                🎉 no goals
              -/
        property := fun f => by
          simp_rw [mem_disjiUnion, mem_antidiagonal, mem_map, Embedding.coeFn_mk, Prod.exists,
            (aux d _).prop, Fin.sum_univ_succ]
          /-
            ι : Type u_1
            μ : Type u_2
            μ' : Type u_3
            inst✝³ : DecidableEq ι
            inst✝² : AddCommMonoid μ
            inst✝¹ : Finset.HasAntidiagonal μ
            inst✝ : DecidableEq μ
            n✝¹ : μ
            d✝¹ : Nat
            n✝ : μ
            d✝ : Nat
            n : μ
            d : Nat
            f : Fin (HAdd.hAdd d 1) → μ
            ⊢ Iff (Exists fun a => Exists fun b => And (Eq (HAdd.hAdd a b) n) (Exists fun  …
          -/
          constructor
            /-
              case mp
              ι : Type u_1
              μ : Type u_2
              μ' : Type u_3
              inst✝³ : DecidableEq ι
              inst✝² : AddCommMonoid μ
              inst✝¹ : Finset.HasAntidiagonal μ
              inst✝ : DecidableEq μ
              n✝¹ : μ
              d✝¹ : Nat
              n✝ : μ
              d✝ : Nat
              n : μ
              d : Nat
              f : Fin (HAdd.hAdd d 1) → μ
              ⊢ (Exists fun a => Exists fun b => And (Eq (HAdd.hAdd a b) n) (Exists fun a_1  …
            -/
          · rintro ⟨a, b, rfl, g, rfl, rfl⟩
            /-
              case mp.intro.intro.intro.intro.intro
              ι : Type u_1
              μ : Type u_2
              μ' : Type u_3
              inst✝³ : DecidableEq ι
              inst✝² : AddCommMonoid μ
              inst✝¹ : Finset.HasAntidiagonal μ
              inst✝ : DecidableEq μ
              n✝ : μ
              d✝¹ : Nat
              n : μ
              d✝ d : Nat
              a : μ
              g : Fin d → μ
              ⊢ Eq (HAdd.hAdd (Fin.cons a g 0) (Finset.univ.sum fun i => Fin.cons a g i.succ …
            -/
            simp only [Fin.cons_zero, Fin.cons_succ]
            /-
              🎉 no goals
            -/
            /-
              case mpr
              ι : Type u_1
              μ : Type u_2
              μ' : Type u_3
              inst✝³ : DecidableEq ι
              inst✝² : AddCommMonoid μ
              inst✝¹ : Finset.HasAntidiagonal μ
              inst✝ : DecidableEq μ
              n✝¹ : μ
              d✝¹ : Nat
              n✝ : μ
              d✝ : Nat
              n : μ
              d : Nat
              f : Fin (HAdd.hAdd d 1) → μ
              ⊢ Eq (HAdd.hAdd (f 0) (Finset.univ.sum fun i => f i.succ)) n → Exists fun a => …
            -/
          · intro hf
            /-
              case mpr
              ι : Type u_1
              μ : Type u_2
              μ' : Type u_3
              inst✝³ : DecidableEq ι
              inst✝² : AddCommMonoid μ
              inst✝¹ : Finset.HasAntidiagonal μ
              inst✝ : DecidableEq μ
              n✝¹ : μ
              d✝¹ : Nat
              n✝ : μ
              d✝ : Nat
              n : μ
              d : Nat
              f : Fin (HAdd.hAdd d 1) → μ
              hf : Eq (HAdd.hAdd (f 0) (Finset.univ.sum fun i => f i.succ)) n
              ⊢ Exists fun a => Exists fun b => And (Eq (HAdd.hAdd a b) n) (Exists fun a_1 = …
            -/
            exact ⟨_, _, hf, _, rfl, Fin.cons_self_tail f⟩ }
            /-
              🎉 no goals
            -/


@[simp] lemma mem_finAntidiagonal {d : ℕ} {f : Fin d → μ} :
    f ∈ finAntidiagonal d n ↔ ∑ i, f i = n := (finAntidiagonal.aux d n).prop f


/-- The finset of functions `ι → μ` with support contained in `s` and sum `n`. -/
def piAntidiag (s : Finset ι) (n : μ) : Finset (ι → μ) := by
  refine (Fintype.truncEquivFinOfCardEq <| Fintype.card_coe s).lift
    (fun e ↦ (finAntidiagonal s.card n).map ⟨fun f i ↦ if hi : i ∈ s then f (e ⟨i, hi⟩) else 0, ?_⟩)
    fun e₁ e₂ ↦ ?_
    /-
      case refine_1
      ι : Type u_1
      μ : Type u_2
      μ' : Type u_3
      inst✝³ : DecidableEq ι
      inst✝² : AddCommMonoid μ
      inst✝¹ : Finset.HasAntidiagonal μ
      inst✝ : DecidableEq μ
      n✝ : μ
      s : Finset ι
      n : μ
      e : Equiv (Subtype fun x => Membership.mem s x) (Fin s.card)
      ⊢ Function.Injective fun f i => dite (Membership.mem s i) (fun hi => f (e ⟨i,  …
    -/
  · rintro f g hfg
    /-
      case refine_1
      ι : Type u_1
      μ : Type u_2
      μ' : Type u_3
      inst✝³ : DecidableEq ι
      inst✝² : AddCommMonoid μ
      inst✝¹ : Finset.HasAntidiagonal μ
      inst✝ : DecidableEq μ
      n✝ : μ
      s : Finset ι
      n : μ
      e : Equiv (Subtype fun x => Membership.mem s x) (Fin s.card)
      f g : Fin s.card → μ
      hfg : Eq ((fun f i => dite (Membership.mem s i) (fun hi => f (e ⟨i, hi⟩)) fun  …
      ⊢ Eq f g
    -/
    ext i
    /-
      case refine_1.h
      ι : Type u_1
      μ : Type u_2
      μ' : Type u_3
      inst✝³ : DecidableEq ι
      inst✝² : AddCommMonoid μ
      inst✝¹ : Finset.HasAntidiagonal μ
      inst✝ : DecidableEq μ
      n✝ : μ
      s : Finset ι
      n : μ
      e : Equiv (Subtype fun x => Membership.mem s x) (Fin s.card)
      f g : Fin s.card → μ
      hfg : Eq ((fun f i => dite (Membership.mem s i) (fun hi => f (e ⟨i, hi⟩)) fun  …
      i : Fin s.card
      ⊢ Eq (f i) (g i)
    -/
    simpa using congr_fun hfg (e.symm i)
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      ι : Type u_1
      μ : Type u_2
      μ' : Type u_3
      inst✝³ : DecidableEq ι
      inst✝² : AddCommMonoid μ
      inst✝¹ : Finset.HasAntidiagonal μ
      inst✝ : DecidableEq μ
      n✝ : μ
      s : Finset ι
      n : μ
      e₁ e₂ : Equiv (Subtype fun x => Membership.mem s x) (Fin s.card)
      ⊢ Eq ((fun e => Finset.map { toFun := fun f i => dite (Membership.mem s i) (fu …
    -/
  · ext f
    /-
      case refine_2.h
      ι : Type u_1
      μ : Type u_2
      μ' : Type u_3
      inst✝³ : DecidableEq ι
      inst✝² : AddCommMonoid μ
      inst✝¹ : Finset.HasAntidiagonal μ
      inst✝ : DecidableEq μ
      n✝ : μ
      s : Finset ι
      n : μ
      e₁ e₂ : Equiv (Subtype fun x => Membership.mem s x) (Fin s.card)
      f : ι → μ
      ⊢ Iff (Membership.mem ((fun e => Finset.map { toFun := fun f i => dite (Member …
    -/
    simp only [mem_map, mem_finAntidiagonal]
    /-
      case refine_2.h
      ι : Type u_1
      μ : Type u_2
      μ' : Type u_3
      inst✝³ : DecidableEq ι
      inst✝² : AddCommMonoid μ
      inst✝¹ : Finset.HasAntidiagonal μ
      inst✝ : DecidableEq μ
      n✝ : μ
      s : Finset ι
      n : μ
      e₁ e₂ : Equiv (Subtype fun x => Membership.mem s x) (Fin s.card)
      f : ι → μ
      ⊢ Iff (Exists fun a => And (Eq (Finset.univ.sum fun i => a i) n) (Eq ({ toFun  …
    -/
    refine Equiv.exists_congr ((e₁.symm.trans e₂).arrowCongr <| .refl _) fun g ↦ ?_
    /-
      case refine_2.h
      ι : Type u_1
      μ : Type u_2
      μ' : Type u_3
      inst✝³ : DecidableEq ι
      inst✝² : AddCommMonoid μ
      inst✝¹ : Finset.HasAntidiagonal μ
      inst✝ : DecidableEq μ
      n✝ : μ
      s : Finset ι
      n : μ
      e₁ e₂ : Equiv (Subtype fun x => Membership.mem s x) (Fin s.card)
      f : ι → μ
      g : Fin s.card → μ
      ⊢ Iff (And (Eq (Finset.univ.sum fun i => g i) n) (Eq ({ toFun := fun f i => di …
    -/
    have := Fintype.sum_equiv (e₂.symm.trans e₁) _ g fun _ ↦ rfl
    /-
      case refine_2.h
      ι : Type u_1
      μ : Type u_2
      μ' : Type u_3
      inst✝³ : DecidableEq ι
      inst✝² : AddCommMonoid μ
      inst✝¹ : Finset.HasAntidiagonal μ
      inst✝ : DecidableEq μ
      n✝ : μ
      s : Finset ι
      n : μ
      e₁ e₂ : Equiv (Subtype fun x => Membership.mem s x) (Fin s.card)
      f : ι → μ
      g : Fin s.card → μ
      this : Eq (Finset.univ.sum fun x => g ((e₂.symm.trans e₁) x)) (Finset.univ.sum …
      ⊢ Iff (And (Eq (Finset.univ.sum fun i => g i) n) (Eq ({ toFun := fun f i => di …
    -/
    aesop
    /-
      🎉 no goals
    -/


@[simp] lemma mem_piAntidiag : f ∈ piAntidiag s n ↔ s.sum f = n ∧ ∀ i, f i ≠ 0 → i ∈ s := by
  /-
    ι : Type u_1
    μ : Type u_2
    inst✝³ : DecidableEq ι
    inst✝² : AddCommMonoid μ
    inst✝¹ : Finset.HasAntidiagonal μ
    inst✝ : DecidableEq μ
    s : Finset ι
    n : μ
    f : ι → μ
    ⊢ Iff (Membership.mem (s.piAntidiag n) f) (And (Eq (s.sum f) n) (∀ (i : ι), Ne …
  -/
  rw [piAntidiag]
  /-
    ι : Type u_1
    μ : Type u_2
    inst✝³ : DecidableEq ι
    inst✝² : AddCommMonoid μ
    inst✝¹ : Finset.HasAntidiagonal μ
    inst✝ : DecidableEq μ
    s : Finset ι
    n : μ
    f : ι → μ
    ⊢ Iff (Membership.mem (Trunc.lift (fun e => Finset.map { toFun := fun f i => d …
  -/
  induction' Fintype.truncEquivFinOfCardEq (Fintype.card_coe s) using Trunc.ind with e
  /-
    case a
    ι : Type u_1
    μ : Type u_2
    inst✝³ : DecidableEq ι
    inst✝² : AddCommMonoid μ
    inst✝¹ : Finset.HasAntidiagonal μ
    inst✝ : DecidableEq μ
    s : Finset ι
    n : μ
    f : ι → μ
    e : Equiv (Subtype fun x => Membership.mem s x) (Fin s.card)
    ⊢ Iff (Membership.mem (Trunc.lift (fun e => Finset.map { toFun := fun f i => d …
  -/
  simp only [Trunc.lift_mk, mem_map, mem_finAntidiagonal, Embedding.coeFn_mk]
  /-
    case a
    ι : Type u_1
    μ : Type u_2
    inst✝³ : DecidableEq ι
    inst✝² : AddCommMonoid μ
    inst✝¹ : Finset.HasAntidiagonal μ
    inst✝ : DecidableEq μ
    s : Finset ι
    n : μ
    f : ι → μ
    e : Equiv (Subtype fun x => Membership.mem s x) (Fin s.card)
    ⊢ Iff (Exists fun a => And (Eq (Finset.univ.sum fun i => a i) n) (Eq (fun i => …
  -/
  constructor
    /-
      case a.mp
      ι : Type u_1
      μ : Type u_2
      inst✝³ : DecidableEq ι
      inst✝² : AddCommMonoid μ
      inst✝¹ : Finset.HasAntidiagonal μ
      inst✝ : DecidableEq μ
      s : Finset ι
      n : μ
      f : ι → μ
      e : Equiv (Subtype fun x => Membership.mem s x) (Fin s.card)
      ⊢ (Exists fun a => And (Eq (Finset.univ.sum fun i => a i) n) (Eq (fun i => dit …
    -/
  · rintro ⟨f, ⟨hf, rfl⟩, rfl⟩
    /-
      case a.mp.intro.intro.refl
      ι : Type u_1
      μ : Type u_2
      inst✝³ : DecidableEq ι
      inst✝² : AddCommMonoid μ
      inst✝¹ : Finset.HasAntidiagonal μ
      inst✝ : DecidableEq μ
      s : Finset ι
      e : Equiv (Subtype fun x => Membership.mem s x) (Fin s.card)
      f : Fin s.card → μ
      ⊢ And (Eq (s.sum fun i => dite (Membership.mem s i) (fun hi => f (e ⟨i, hi⟩))  …
    -/
    rw [sum_dite_of_true fun _ ↦ id]
    /-
      case a.mp.intro.intro.refl
      ι : Type u_1
      μ : Type u_2
      inst✝³ : DecidableEq ι
      inst✝² : AddCommMonoid μ
      inst✝¹ : Finset.HasAntidiagonal μ
      inst✝ : DecidableEq μ
      s : Finset ι
      e : Equiv (Subtype fun x => Membership.mem s x) (Fin s.card)
      f : Fin s.card → μ
      ⊢ And (Eq (Finset.univ.sum fun i => f (e ⟨↑i, ⋯⟩)) (Finset.univ.sum fun i => f …
    -/
    exact ⟨Fintype.sum_equiv e _ _ (by simp), by simp +contextual⟩
    /-
      🎉 no goals
    -/
    /-
      case a.mpr
      ι : Type u_1
      μ : Type u_2
      inst✝³ : DecidableEq ι
      inst✝² : AddCommMonoid μ
      inst✝¹ : Finset.HasAntidiagonal μ
      inst✝ : DecidableEq μ
      s : Finset ι
      n : μ
      f : ι → μ
      e : Equiv (Subtype fun x => Membership.mem s x) (Fin s.card)
      ⊢ And (Eq (s.sum f) n) (∀ (i : ι), Ne (f i) 0 → Membership.mem s i) → Exists f …
    -/
  · rintro ⟨rfl, hf⟩
    /-
      case a.mpr.intro
      ι : Type u_1
      μ : Type u_2
      inst✝³ : DecidableEq ι
      inst✝² : AddCommMonoid μ
      inst✝¹ : Finset.HasAntidiagonal μ
      inst✝ : DecidableEq μ
      s : Finset ι
      f : ι → μ
      e : Equiv (Subtype fun x => Membership.mem s x) (Fin s.card)
      hf : ∀ (i : ι), Ne (f i) 0 → Membership.mem s i
      ⊢ Exists fun a => And (Eq (Finset.univ.sum fun i => a i) (s.sum f)) (Eq (fun i …
    -/
    refine ⟨f ∘ (↑) ∘ e.symm, ?_, by ext i; have := not_imp_comm.1 (hf i); aesop⟩
    /-
      case a.mpr.intro
      ι : Type u_1
      μ : Type u_2
      inst✝³ : DecidableEq ι
      inst✝² : AddCommMonoid μ
      inst✝¹ : Finset.HasAntidiagonal μ
      inst✝ : DecidableEq μ
      s : Finset ι
      f : ι → μ
      e : Equiv (Subtype fun x => Membership.mem s x) (Fin s.card)
      hf : ∀ (i : ι), Ne (f i) 0 → Membership.mem s i
      ⊢ Eq (Finset.univ.sum fun i => Function.comp f (Function.comp Subtype.val ⇑e.s …
    -/
    rw [← sum_attach s]
    /-
      case a.mpr.intro
      ι : Type u_1
      μ : Type u_2
      inst✝³ : DecidableEq ι
      inst✝² : AddCommMonoid μ
      inst✝¹ : Finset.HasAntidiagonal μ
      inst✝ : DecidableEq μ
      s : Finset ι
      f : ι → μ
      e : Equiv (Subtype fun x => Membership.mem s x) (Fin s.card)
      hf : ∀ (i : ι), Ne (f i) 0 → Membership.mem s i
      ⊢ Eq (Finset.univ.sum fun i => Function.comp f (Function.comp Subtype.val ⇑e.s …
    -/
    exact Fintype.sum_equiv e.symm _ _ (by simp)
    /-
      🎉 no goals
    -/


@[simp] lemma piAntidiag_empty_zero : piAntidiag (∅ : Finset ι) (0 : μ) = {0} := by
  /-
    ι : Type u_1
    μ : Type u_2
    inst✝³ : DecidableEq ι
    inst✝² : AddCommMonoid μ
    inst✝¹ : Finset.HasAntidiagonal μ
    inst✝ : DecidableEq μ
    ⊢ Eq (EmptyCollection.emptyCollection.piAntidiag 0) (Singleton.singleton 0)
  -/
  ext; simp [Fintype.sum_eq_zero_iff_of_nonneg, funext_iff, not_imp_comm, ← forall_and]
       /-
         🎉 no goals
       -/


@[simp] lemma piAntidiag_empty_of_ne_zero (hn : n ≠ 0) : piAntidiag (∅ : Finset ι) n = ∅ :=
                                 /-
                                   ι : Type u_1
                                   μ : Type u_2
                                   inst✝³ : DecidableEq ι
                                   inst✝² : AddCommMonoid μ
                                   inst✝¹ : Finset.HasAntidiagonal μ
                                   inst✝ : DecidableEq μ
                                   n : μ
                                   hn : Ne n 0
                                   ⊢ ∀ (x : ι → μ), Not (Membership.mem (EmptyCollection.emptyCollection.piAntidi …
                                 -/
  eq_empty_of_forall_not_mem (by simp [@eq_comm _ 0, hn.symm])
                                 /-
                                   🎉 no goals
                                 -/


lemma piAntidiag_empty (n : μ) : piAntidiag (∅ : Finset ι) n = if n = 0 then {0} else ∅ := by
  /-
    ι : Type u_1
    μ : Type u_2
    inst✝³ : DecidableEq ι
    inst✝² : AddCommMonoid μ
    inst✝¹ : Finset.HasAntidiagonal μ
    inst✝ : DecidableEq μ
    n : μ
    ⊢ Eq (EmptyCollection.emptyCollection.piAntidiag n) (ite (Eq n 0) (Singleton.s …
  -/
                        /-
                          🎉 no goals
                        -/
  split_ifs with hn <;> simp [*]
                        /-
                          🎉 no goals
                        -/


lemma finsetCongr_piAntidiag_eq_antidiag (n : μ) :
    Equiv.finsetCongr (Equiv.boolArrowEquivProd _) (piAntidiag univ n) = antidiagonal n := by
  /-
    μ : Type u_2
    inst✝² : AddCommMonoid μ
    inst✝¹ : Finset.HasAntidiagonal μ
    inst✝ : DecidableEq μ
    n : μ
    ⊢ Eq ((Equiv.boolArrowEquivProd μ).finsetCongr (Finset.univ.piAntidiag n)) (Fi …
  -/
  ext ⟨x₁, x₂⟩
  simp_rw [Equiv.finsetCongr_apply, mem_map, Equiv.toEmbedding, Function.Embedding.coeFn_mk,
    ← Equiv.eq_symm_apply]
  /-
    case h.mk
    μ : Type u_2
    inst✝² : AddCommMonoid μ
    inst✝¹ : Finset.HasAntidiagonal μ
    inst✝ : DecidableEq μ
    n x₁ x₂ : μ
    ⊢ Iff (Exists fun a => And (Membership.mem (Finset.univ.piAntidiag n) a) (Eq a …
  -/
  simp [add_comm]
  /-
    🎉 no goals
  -/


lemma pairwiseDisjoint_piAntidiag_map_addRightEmbedding (hi : i ∉ s) (n : μ) :
    (antidiagonal n : Set (μ × μ)).PairwiseDisjoint fun p ↦
      map (addRightEmbedding fun j ↦ if j = i then p.1 else 0) (s.piAntidiag p.2) := by
  /-
    ι : Type u_1
    μ : Type u_2
    inst✝³ : DecidableEq ι
    inst✝² : AddCancelCommMonoid μ
    inst✝¹ : Finset.HasAntidiagonal μ
    inst✝ : DecidableEq μ
    i : ι
    s : Finset ι
    hi : Not (Membership.mem s i)
    n : μ
    ⊢ (↑(Finset.HasAntidiagonal.antidiagonal n)).PairwiseDisjoint fun p => Finset. …
  -/
  rintro ⟨a, b⟩ hab ⟨c, d⟩ hcd
  simp only [ne_eq, antidiagonal_congr' hab hcd, disjoint_left, mem_map, mem_piAntidiag,
    addRightEmbedding_apply, not_exists, not_and, and_imp, forall_exists_index]
  /-
    case mk.mk
    ι : Type u_1
    μ : Type u_2
    inst✝³ : DecidableEq ι
    inst✝² : AddCancelCommMonoid μ
    inst✝¹ : Finset.HasAntidiagonal μ
    inst✝ : DecidableEq μ
    i : ι
    s : Finset ι
    hi : Not (Membership.mem s i)
    n a b : μ
    hab : Membership.mem ↑(Finset.HasAntidiagonal.antidiagonal n) { fst := a, snd  …
    c d : μ
    hcd : Membership.mem ↑(Finset.HasAntidiagonal.antidiagonal n) { fst := c, snd  …
    ⊢ Not (Eq b d) → ∀ ⦃a_2 : ι → μ⦄ (x : ι → μ), Eq (s.sum x) b → (∀ (i : ι), Not …
  -/
  rintro hfg _ f rfl - rfl g rfl - hgf
  /-
    case mk.mk
    ι : Type u_1
    μ : Type u_2
    inst✝³ : DecidableEq ι
    inst✝² : AddCancelCommMonoid μ
    inst✝¹ : Finset.HasAntidiagonal μ
    inst✝ : DecidableEq μ
    i : ι
    s : Finset ι
    hi : Not (Membership.mem s i)
    n a c : μ
    f : ι → μ
    hab : Membership.mem ↑(Finset.HasAntidiagonal.antidiagonal n) { fst := a, snd  …
    g : ι → μ
    hcd : Membership.mem ↑(Finset.HasAntidiagonal.antidiagonal n) { fst := c, snd  …
    hfg : Not (Eq (s.sum f) (s.sum g))
    hgf : Eq (HAdd.hAdd g fun j => ite (Eq j i) c 0) (HAdd.hAdd f fun j => ite (Eq …
    ⊢ False
  -/
  exact hfg <| by simpa [sum_add_distrib, hi] using congr_arg (∑ j ∈ s, · j) hgf.symm
  /-
    🎉 no goals
  -/


lemma piAntidiag_cons (hi : i ∉ s) (n : μ) :
    piAntidiag (cons i s hi) n = (antidiagonal n).disjiUnion (fun p : μ × μ ↦
      (piAntidiag s p.snd).map (addRightEmbedding fun t ↦ if t = i then p.fst else 0))
        (pairwiseDisjoint_piAntidiag_map_addRightEmbedding hi _) := by
  /-
    ι : Type u_1
    μ : Type u_2
    inst✝³ : DecidableEq ι
    inst✝² : AddCancelCommMonoid μ
    inst✝¹ : Finset.HasAntidiagonal μ
    inst✝ : DecidableEq μ
    i : ι
    s : Finset ι
    hi : Not (Membership.mem s i)
    n : μ
    ⊢ Eq ((Finset.cons i s hi).piAntidiag n) ((Finset.HasAntidiagonal.antidiagonal …
  -/
  ext f
  simp only [mem_piAntidiag, sum_cons, ne_eq, mem_cons, mem_disjiUnion, mem_antidiagonal, mem_map,
    addLeftEmbedding_apply, Prod.exists]
  /-
    case h
    ι : Type u_1
    μ : Type u_2
    inst✝³ : DecidableEq ι
    inst✝² : AddCancelCommMonoid μ
    inst✝¹ : Finset.HasAntidiagonal μ
    inst✝ : DecidableEq μ
    i : ι
    s : Finset ι
    hi : Not (Membership.mem s i)
    n : μ
    f : ι → μ
    ⊢ Iff (And (Eq (HAdd.hAdd (f i) (s.sum fun x => f x)) n) (∀ (i_1 : ι), Not (Eq …
  -/
  constructor
    /-
      case h.mp
      ι : Type u_1
      μ : Type u_2
      inst✝³ : DecidableEq ι
      inst✝² : AddCancelCommMonoid μ
      inst✝¹ : Finset.HasAntidiagonal μ
      inst✝ : DecidableEq μ
      i : ι
      s : Finset ι
      hi : Not (Membership.mem s i)
      n : μ
      f : ι → μ
      ⊢ And (Eq (HAdd.hAdd (f i) (s.sum fun x => f x)) n) (∀ (i_1 : ι), Not (Eq (f i …
    -/
  · rintro ⟨hn, hf⟩
    /-
      case h.mp.intro
      ι : Type u_1
      μ : Type u_2
      inst✝³ : DecidableEq ι
      inst✝² : AddCancelCommMonoid μ
      inst✝¹ : Finset.HasAntidiagonal μ
      inst✝ : DecidableEq μ
      i : ι
      s : Finset ι
      hi : Not (Membership.mem s i)
      n : μ
      f : ι → μ
      hn : Eq (HAdd.hAdd (f i) (s.sum fun x => f x)) n
      hf : ∀ (i_1 : ι), Not (Eq (f i_1) 0) → Or (Eq i_1 i) (Membership.mem s i_1)
      ⊢ Exists fun a => Exists fun b => And (Eq (HAdd.hAdd a b) n) (Exists fun a_1 = …
    -/
    refine ⟨_, _, hn, update f i 0, ⟨sum_update_of_not_mem hi _ _, fun j ↦ ?_⟩, by aesop⟩
    /-
      case h.mp.intro
      ι : Type u_1
      μ : Type u_2
      inst✝³ : DecidableEq ι
      inst✝² : AddCancelCommMonoid μ
      inst✝¹ : Finset.HasAntidiagonal μ
      inst✝ : DecidableEq μ
      i : ι
      s : Finset ι
      hi : Not (Membership.mem s i)
      n : μ
      f : ι → μ
      hn : Eq (HAdd.hAdd (f i) (s.sum fun x => f x)) n
      hf : ∀ (i_1 : ι), Not (Eq (f i_1) 0) → Or (Eq i_1 i) (Membership.mem s i_1)
      j : ι
      ⊢ Not (Eq (Function.update f i 0 j) 0) → Membership.mem s j
    -/
    have := fun h₁ h₂ ↦ (hf j h₁).resolve_left h₂
    /-
      case h.mp.intro
      ι : Type u_1
      μ : Type u_2
      inst✝³ : DecidableEq ι
      inst✝² : AddCancelCommMonoid μ
      inst✝¹ : Finset.HasAntidiagonal μ
      inst✝ : DecidableEq μ
      i : ι
      s : Finset ι
      hi : Not (Membership.mem s i)
      n : μ
      f : ι → μ
      hn : Eq (HAdd.hAdd (f i) (s.sum fun x => f x)) n
      hf : ∀ (i_1 : ι), Not (Eq (f i_1) 0) → Or (Eq i_1 i) (Membership.mem s i_1)
      j : ι
      this : Not (Eq (f j) 0) → Not (Eq j i) → Membership.mem s j
      ⊢ Not (Eq (Function.update f i 0 j) 0) → Membership.mem s j
    -/
    aesop (add simp [update])
    /-
      🎉 no goals
    -/
    /-
      case h.mpr
      ι : Type u_1
      μ : Type u_2
      inst✝³ : DecidableEq ι
      inst✝² : AddCancelCommMonoid μ
      inst✝¹ : Finset.HasAntidiagonal μ
      inst✝ : DecidableEq μ
      i : ι
      s : Finset ι
      hi : Not (Membership.mem s i)
      n : μ
      f : ι → μ
      ⊢ (Exists fun a => Exists fun b => And (Eq (HAdd.hAdd a b) n) (Exists fun a_1  …
    -/
  · rintro ⟨a, _, hn, g, ⟨rfl, hg⟩, rfl⟩
    /-
      case h.mpr.intro.intro.intro.intro.intro.intro
      ι : Type u_1
      μ : Type u_2
      inst✝³ : DecidableEq ι
      inst✝² : AddCancelCommMonoid μ
      inst✝¹ : Finset.HasAntidiagonal μ
      inst✝ : DecidableEq μ
      i : ι
      s : Finset ι
      hi : Not (Membership.mem s i)
      n a : μ
      g : ι → μ
      hg : ∀ (i : ι), Not (Eq (g i) 0) → Membership.mem s i
      hn : Eq (HAdd.hAdd a (s.sum g)) n
      ⊢ And (Eq (HAdd.hAdd ((addRightEmbedding fun t => ite (Eq t i) a 0) g i) (s.su …
    -/
    have := hg i
    /-
      case h.mpr.intro.intro.intro.intro.intro.intro
      ι : Type u_1
      μ : Type u_2
      inst✝³ : DecidableEq ι
      inst✝² : AddCancelCommMonoid μ
      inst✝¹ : Finset.HasAntidiagonal μ
      inst✝ : DecidableEq μ
      i : ι
      s : Finset ι
      hi : Not (Membership.mem s i)
      n a : μ
      g : ι → μ
      hg : ∀ (i : ι), Not (Eq (g i) 0) → Membership.mem s i
      hn : Eq (HAdd.hAdd a (s.sum g)) n
      this : Not (Eq (g i) 0) → Membership.mem s i
      ⊢ And (Eq (HAdd.hAdd ((addRightEmbedding fun t => ite (Eq t i) a 0) g i) (s.su …
    -/
    aesop (add simp [sum_add_distrib])
    /-
      🎉 no goals
    -/


lemma piAntidiag_insert [DecidableEq (ι → μ)] (hi : i ∉ s) (n : μ) :
    piAntidiag (insert i s) n = (antidiagonal n).biUnion fun p : μ × μ ↦ (piAntidiag s p.snd).image
      (fun f j ↦ f j + if j = i then p.fst else 0) := by
  /-
    ι : Type u_1
    μ : Type u_2
    inst✝⁴ : DecidableEq ι
    inst✝³ : AddCancelCommMonoid μ
    inst✝² : Finset.HasAntidiagonal μ
    inst✝¹ : DecidableEq μ
    i : ι
    s : Finset ι
    inst✝ : DecidableEq (ι → μ)
    hi : Not (Membership.mem s i)
    n : μ
    ⊢ Eq ((Insert.insert i s).piAntidiag n) ((Finset.HasAntidiagonal.antidiagonal  …
  -/
  simpa [map_eq_image, addRightEmbedding] using piAntidiag_cons hi n
  /-
    🎉 no goals
  -/


@[simp] lemma piAntidiag_zero (s : Finset ι) : piAntidiag s (0 : μ) = {0} := by
  /-
    ι : Type u_1
    μ : Type u_2
    inst✝³ : DecidableEq ι
    inst✝² : CanonicallyOrderedAddCommMonoid μ
    inst✝¹ : Finset.HasAntidiagonal μ
    inst✝ : DecidableEq μ
    s : Finset ι
    ⊢ Eq (s.piAntidiag 0) (Singleton.singleton 0)
  -/
  ext; simp [Fintype.sum_eq_zero_iff_of_nonneg, funext_iff, not_imp_comm, ← forall_and]
       /-
         🎉 no goals
       -/


/-- Local notation for the pointwise operation `n • s := {n • a | a ∈ s}` to avoid conflict with the
pointwise operation `n • s := s + ... + s` (`n` times). -/
local infixr:73 "•ℕ" => @SMul.smul _ _ Finset.smulFinset


lemma piAntidiag_univ_fin_eq_antidiagonalTuple (n k : ℕ) :
    piAntidiag univ n = Nat.antidiagonalTuple k n := by
  /-
    n k : Nat
    ⊢ Eq (Finset.univ.piAntidiag n) (Finset.Nat.antidiagonalTuple k n)
  -/
  ext; simp [Nat.mem_antidiagonalTuple]
       /-
         🎉 no goals
       -/


lemma nsmul_piAntidiag [DecidableEq (ι → ℕ)] (s : Finset ι) (m : ℕ) {n : ℕ} (hn : n ≠ 0) :
    n •ℕ piAntidiag s m = (piAntidiag s (n * m)).filter fun f : ι → ℕ ↦ ∀ i ∈ s, n ∣ f i := by
  /-
    ι : Type u_1
    inst✝¹ : DecidableEq ι
    inst✝ : DecidableEq (ι → Nat)
    s : Finset ι
    m n : Nat
    hn : Ne n 0
    ⊢ Eq (SMul.smul n (s.piAntidiag m)) (Finset.filter (fun f => ∀ (i : ι), Member …
  -/
  ext f
  /-
    case h
    ι : Type u_1
    inst✝¹ : DecidableEq ι
    inst✝ : DecidableEq (ι → Nat)
    s : Finset ι
    m n : Nat
    hn : Ne n 0
    f : ι → Nat
    ⊢ Iff (Membership.mem (SMul.smul n (s.piAntidiag m)) f) (Membership.mem (Finse …
  -/
  refine mem_smul_finset.trans ?_
  simp only [mem_smul_finset, mem_filter, mem_piAntidiag, Function.Embedding.coeFn_mk, exists_prop,
    and_assoc]
  /-
    case h
    ι : Type u_1
    inst✝¹ : DecidableEq ι
    inst✝ : DecidableEq (ι → Nat)
    s : Finset ι
    m n : Nat
    hn : Ne n 0
    f : ι → Nat
    ⊢ Iff (Exists fun y => And (Eq (s.sum y) m) (And (∀ (i : ι), Ne (y i) 0 → Memb …
  -/
  constructor
    /-
      case h.mp
      ι : Type u_1
      inst✝¹ : DecidableEq ι
      inst✝ : DecidableEq (ι → Nat)
      s : Finset ι
      m n : Nat
      hn : Ne n 0
      f : ι → Nat
      ⊢ (Exists fun y => And (Eq (s.sum y) m) (And (∀ (i : ι), Ne (y i) 0 → Membersh …
    -/
  · rintro ⟨f, rfl, hf, rfl⟩
    /-
      case h.mp.intro.intro.intro
      ι : Type u_1
      inst✝¹ : DecidableEq ι
      inst✝ : DecidableEq (ι → Nat)
      s : Finset ι
      n : Nat
      hn : Ne n 0
      f : ι → Nat
      hf : ∀ (i : ι), Ne (f i) 0 → Membership.mem s i
      ⊢ And (Eq (s.sum (HSMul.hSMul n f)) (HMul.hMul n (s.sum f))) (And (∀ (i : ι),  …
    -/
    simpa [← mul_sum, hn] using hf
    /-
      🎉 no goals
    -/
  /-
    case h.mpr
    ι : Type u_1
    inst✝¹ : DecidableEq ι
    inst✝ : DecidableEq (ι → Nat)
    s : Finset ι
    m n : Nat
    hn : Ne n 0
    f : ι → Nat
    ⊢ And (Eq (s.sum f) (HMul.hMul n m)) (And (∀ (i : ι), Ne (f i) 0 → Membership. …
  -/
  rintro ⟨hfsum, hfsup, hfdvd⟩
  have (i) : n ∣ f i := by
    by_cases hi : i ∈ s
    · exact hfdvd _ hi
    · rw [not_imp_comm.1 (hfsup _) hi]
      exact dvd_zero _
  /-
    case h.mpr.intro.intro
    ι : Type u_1
    inst✝¹ : DecidableEq ι
    inst✝ : DecidableEq (ι → Nat)
    s : Finset ι
    m n : Nat
    hn : Ne n 0
    f : ι → Nat
    hfsum : Eq (s.sum f) (HMul.hMul n m)
    hfsup : ∀ (i : ι), Ne (f i) 0 → Membership.mem s i
    hfdvd : ∀ (i : ι), Membership.mem s i → Dvd.dvd n (f i)
    this : ∀ (i : ι), Dvd.dvd n (f i)
    ⊢ Exists fun y => And (Eq (s.sum y) m) (And (∀ (i : ι), Ne (y i) 0 → Membershi …
  -/
  refine ⟨fun i ↦ f i / n, ?_⟩
  /-
    case h.mpr.intro.intro
    ι : Type u_1
    inst✝¹ : DecidableEq ι
    inst✝ : DecidableEq (ι → Nat)
    s : Finset ι
    m n : Nat
    hn : Ne n 0
    f : ι → Nat
    hfsum : Eq (s.sum f) (HMul.hMul n m)
    hfsup : ∀ (i : ι), Ne (f i) 0 → Membership.mem s i
    hfdvd : ∀ (i : ι), Membership.mem s i → Dvd.dvd n (f i)
    this : ∀ (i : ι), Dvd.dvd n (f i)
    ⊢ And (Eq (s.sum fun i => HDiv.hDiv (f i) n) m) (And (∀ (i : ι), Ne ((fun i => …
  -/
  simp [funext_iff, Nat.mul_div_cancel', ← Nat.sum_div, *]
  /-
    case h.mpr.intro.intro
    ι : Type u_1
    inst✝¹ : DecidableEq ι
    inst✝ : DecidableEq (ι → Nat)
    s : Finset ι
    m n : Nat
    hn : Ne n 0
    f : ι → Nat
    hfsum : Eq (s.sum f) (HMul.hMul n m)
    hfsup : ∀ (i : ι), Ne (f i) 0 → Membership.mem s i
    hfdvd : ∀ (i : ι), Membership.mem s i → Dvd.dvd n (f i)
    this : ∀ (i : ι), Dvd.dvd n (f i)
    ⊢ ∀ (i : ι), LE.le n (f i) → Membership.mem s i
  -/
  aesop
  /-
    🎉 no goals
  -/


lemma map_nsmul_piAntidiag (s : Finset ι) (m : ℕ) {n : ℕ} (hn : n ≠ 0) :
    (piAntidiag s m).map
      ⟨(n • ·), fun _ _ h ↦ funext fun i ↦ mul_right_injective₀ hn (congr_fun h i)⟩ =
        (piAntidiag s (n * m)).filter fun f : ι → ℕ ↦ ∀ i ∈ s, n ∣ f i := by
  /-
    ι : Type u_1
    inst✝ : DecidableEq ι
    s : Finset ι
    m n : Nat
    hn : Ne n 0
    ⊢ Eq (Finset.map { toFun := fun x => HSMul.hSMul n x, inj' := ⋯ } (s.piAntidia …
  -/
  classical rw [map_eq_image]; exact nsmul_piAntidiag _ _ hn
  /-
    🎉 no goals
  -/


lemma nsmul_piAntidiag_univ [Fintype ι] (m : ℕ) {n : ℕ} (hn : n ≠ 0) :
    @SMul.smul _ _ Finset.smulFinset n (piAntidiag univ m) =
      (piAntidiag univ (n * m)).filter fun f : ι → ℕ ↦ ∀ i, n ∣ f i := by
  /-
    ι : Type u_1
    inst✝¹ : DecidableEq ι
    inst✝ : Fintype ι
    m n : Nat
    hn : Ne n 0
    ⊢ Eq (SMul.smul n (Finset.univ.piAntidiag m)) (Finset.filter (fun f => ∀ (i :  …
  -/
  simpa using nsmul_piAntidiag (univ : Finset ι) m hn
  /-
    🎉 no goals
  -/


lemma map_nsmul_piAntidiag_univ [Fintype ι] (m : ℕ) {n : ℕ} (hn : n ≠ 0) :
    (piAntidiag (univ : Finset ι) m).map
        ⟨(n • ·), fun _ _ h ↦ funext fun i ↦ mul_right_injective₀ hn (congr_fun h i)⟩ =
      (piAntidiag univ (n * m)).filter fun f : ι → ℕ ↦ ∀ i, n ∣ f i := by
  /-
    ι : Type u_1
    inst✝¹ : DecidableEq ι
    inst✝ : Fintype ι
    m n : Nat
    hn : Ne n 0
    ⊢ Eq (Finset.map { toFun := fun x => HSMul.hSMul n x, inj' := ⋯ } (Finset.univ …
  -/
  simpa using map_nsmul_piAntidiag (univ : Finset ι) m hn
  /-
    🎉 no goals
  -/


lemma map_sym_eq_piAntidiag [DecidableEq ι] (s : Finset ι) (n : ℕ) :
    (s.sym n).map ⟨fun m a ↦ m.1.count a, Multiset.count_injective.comp Sym.coe_injective⟩ =
      piAntidiag s n := by
  /-
    ι : Type u_1
    inst✝ : DecidableEq ι
    s : Finset ι
    n : Nat
    ⊢ Eq (Finset.map { toFun := fun m a => Multiset.count a ↑m, inj' := ⋯ } (s.sym …
  -/
  ext f
  simp only [Sym.val_eq_coe, mem_map, mem_sym_iff, Embedding.coeFn_mk, funext_iff, Sym.exists,
    Sym.mem_mk, Sym.coe_mk, exists_and_left, exists_prop, mem_piAntidiag, ne_eq]
  /-
    case h
    ι : Type u_1
    inst✝ : DecidableEq ι
    s : Finset ι
    n : Nat
    f : ι → Nat
    ⊢ Iff (Exists fun s_1 => And (∀ (a : ι), Membership.mem s_1 a → Membership.mem …
  -/
  constructor
    /-
      case h.mp
      ι : Type u_1
      inst✝ : DecidableEq ι
      s : Finset ι
      n : Nat
      f : ι → Nat
      ⊢ (Exists fun s_1 => And (∀ (a : ι), Membership.mem s_1 a → Membership.mem s a …
    -/
  · rintro ⟨m, hm, rfl, hf⟩
    /-
      case h.mp.intro.intro.intro
      ι : Type u_1
      inst✝ : DecidableEq ι
      s : Finset ι
      f : ι → Nat
      m : Multiset ι
      hm : ∀ (a : ι), Membership.mem m a → Membership.mem s a
      hf : ∀ (x : ι), Eq (Multiset.count x m) (f x)
      ⊢ And (Eq (s.sum f) m.card) (∀ (i : ι), Not (Eq (f i) 0) → Membership.mem s i)
    -/
    simpa [← hf, Multiset.sum_count_eq_card hm]
    /-
      🎉 no goals
    -/
    /-
      case h.mpr
      ι : Type u_1
      inst✝ : DecidableEq ι
      s : Finset ι
      n : Nat
      f : ι → Nat
      ⊢ And (Eq (s.sum f) n) (∀ (i : ι), Not (Eq (f i) 0) → Membership.mem s i) → Ex …
    -/
  · rintro ⟨rfl, hf⟩
    /-
      case h.mpr.intro
      ι : Type u_1
      inst✝ : DecidableEq ι
      s : Finset ι
      f : ι → Nat
      hf : ∀ (i : ι), Not (Eq (f i) 0) → Membership.mem s i
      ⊢ Exists fun s_1 => And (∀ (a : ι), Membership.mem s_1 a → Membership.mem s a) …
    -/
    refine ⟨∑ a ∈ s, f a • {a}, ?_, ?_⟩
      /-
        case h.mpr.intro.refine_1
        ι : Type u_1
        inst✝ : DecidableEq ι
        s : Finset ι
        f : ι → Nat
        hf : ∀ (i : ι), Not (Eq (f i) 0) → Membership.mem s i
        ⊢ ∀ (a : ι), Membership.mem (s.sum fun a => HSMul.hSMul (f a) (Singleton.singl …
      -/
    · simp +contextual
      /-
        🎉 no goals
      -/
      /-
        case h.mpr.intro.refine_2
        ι : Type u_1
        inst✝ : DecidableEq ι
        s : Finset ι
        f : ι → Nat
        hf : ∀ (i : ι), Not (Eq (f i) 0) → Membership.mem s i
        ⊢ And (Eq (s.sum fun a => HSMul.hSMul (f a) (Singleton.singleton a)).card (s.s …
      -/
    · simpa [Multiset.count_sum', Multiset.count_singleton, not_imp_comm, eq_comm (a := 0)] using hf
      /-
        🎉 no goals
      -/


