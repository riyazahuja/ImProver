/-- The antidiagonal of a multiset `s` consists of all pairs `(t₁, t₂)`
    such that `t₁ + t₂ = s`. These pairs are counted with multiplicities. -/
def antidiagonal (s : Multiset α) : Multiset (Multiset α × Multiset α) :=
  Quot.liftOn s (fun l ↦ (revzip (powersetAux l) : Multiset (Multiset α × Multiset α)))
    fun _ _ h ↦ Quot.sound (revzip_powersetAux_perm h)


theorem antidiagonal_coe (l : List α) : @antidiagonal α l = revzip (powersetAux l) :=
  rfl


@[simp]
theorem antidiagonal_coe' (l : List α) : @antidiagonal α l = revzip (powersetAux' l) :=
  Quot.sound revzip_powersetAux_perm_aux'

/- Porting note: `simp` seemed to be applying `antidiagonal_coe'` instead of `antidiagonal_coe`
in what used to be `simp [antidiagonal_coe]`. -/

/-- A pair `(t₁, t₂)` of multisets is contained in `antidiagonal s`
    if and only if `t₁ + t₂ = s`. -/
@[simp]
theorem mem_antidiagonal {s : Multiset α} {x : Multiset α × Multiset α} :
    x ∈ antidiagonal s ↔ x.1 + x.2 = s :=
  Quotient.inductionOn s fun l ↦ by
    /-
      α : Type u_1
      s : Multiset α
      x : Prod (Multiset α) (Multiset α)
      l : List α
      ⊢ Iff (Membership.mem (Multiset.antidiagonal (Quotient.mk (List.isSetoid α) l) …
    -/
    dsimp only [quot_mk_to_coe, antidiagonal_coe]
    /-
      α : Type u_1
      s : Multiset α
      x : Prod (Multiset α) (Multiset α)
      l : List α
      ⊢ Iff (Membership.mem (↑(Multiset.powersetAux l).revzip) x) (Eq (HAdd.hAdd x.1 …
    -/
    refine ⟨fun h => revzip_powersetAux h, fun h ↦ ?_⟩
    /-
      α : Type u_1
      s : Multiset α
      x : Prod (Multiset α) (Multiset α)
      l : List α
      h : Eq (HAdd.hAdd x.1 x.2) ↑l
      ⊢ Membership.mem (↑(Multiset.powersetAux l).revzip) x
    -/
    haveI := Classical.decEq α
    simp only [revzip_powersetAux_lemma l revzip_powersetAux, h.symm, mem_coe,
      List.mem_map, mem_powersetAux]
    /-
      α : Type u_1
      s : Multiset α
      x : Prod (Multiset α) (Multiset α)
      l : List α
      h : Eq (HAdd.hAdd x.1 x.2) ↑l
      this : DecidableEq α
      ⊢ Exists fun a => And (LE.le a (HAdd.hAdd x.1 x.2)) (Eq { fst := a, snd := HSu …
    -/
    cases' x with x₁ x₂
    /-
      case mk
      α : Type u_1
      s : Multiset α
      l : List α
      this : DecidableEq α
      x₁ x₂ : Multiset α
      h : Eq (HAdd.hAdd { fst := x₁, snd := x₂ }.1 { fst := x₁, snd := x₂ }.2) ↑l
      ⊢ Exists fun a => And (LE.le a (HAdd.hAdd { fst := x₁, snd := x₂ }.1 { fst :=  …
    -/
    exact ⟨x₁, le_add_right _ _, by rw [add_tsub_cancel_left x₁ x₂]⟩
    /-
      🎉 no goals
    -/


@[simp]
theorem antidiagonal_map_fst (s : Multiset α) : (antidiagonal s).map Prod.fst = powerset s :=
                                    /-
                                      α : Type u_1
                                      s : Multiset α
                                      l : List α
                                      ⊢ Eq (Multiset.map Prod.fst (Multiset.antidiagonal (Quotient.mk (List.isSetoid …
                                    -/
  Quotient.inductionOn s fun l ↦ by simp [powersetAux']
                                    /-
                                      🎉 no goals
                                    -/


@[simp]
theorem antidiagonal_map_snd (s : Multiset α) : (antidiagonal s).map Prod.snd = powerset s :=
                                    /-
                                      α : Type u_1
                                      s : Multiset α
                                      l : List α
                                      ⊢ Eq (Multiset.map Prod.snd (Multiset.antidiagonal (Quotient.mk (List.isSetoid …
                                    -/
  Quotient.inductionOn s fun l ↦ by simp [powersetAux']
                                    /-
                                      🎉 no goals
                                    -/


@[simp]
theorem antidiagonal_zero : @antidiagonal α 0 = {(0, 0)} :=
  rfl


@[simp]
theorem antidiagonal_cons (a : α) (s) :
    antidiagonal (a ::ₘ s) =
      map (Prod.map id (cons a)) (antidiagonal s) + map (Prod.map (cons a) id) (antidiagonal s) :=
  Quotient.inductionOn s fun l ↦ by
    simp only [revzip, reverse_append, quot_mk_to_coe, coe_eq_coe, powersetAux'_cons, cons_coe,
      map_coe, antidiagonal_coe', coe_add]
    /-
      α : Type u_1
      a : α
      s : Multiset α
      l : List α
      ⊢ ((HAppend.hAppend (Multiset.powersetAux' l) (List.map (Multiset.cons a) (Mul …
    -/
    rw [← zip_map, ← zip_map, zip_append, (_ : _ ++ _ = _)]
      /-
        α : Type u_1
        a : α
        s : Multiset α
        l : List α
        ⊢ Eq (HAppend.hAppend ((Multiset.powersetAux' l).zip (List.map (Multiset.cons  …
      -/
    · congr
        /-
          case e_a.e_a
          α : Type u_1
          a : α
          s : Multiset α
          l : List α
          ⊢ Eq (Multiset.powersetAux' l) (List.map id (Multiset.powersetAux' l))
        -/
      · simp only [List.map_id]
        /-
          🎉 no goals
        -/
        /-
          case e_a.e_a
          α : Type u_1
          a : α
          s : Multiset α
          l : List α
          ⊢ Eq (List.map (Multiset.cons a) (Multiset.powersetAux' l)).reverse (List.map  …
        -/
      · rw [map_reverse]
        /-
          🎉 no goals
        -/
        /-
          case e_a.e_a
          α : Type u_1
          a : α
          s : Multiset α
          l : List α
          ⊢ Eq (Multiset.powersetAux' l).reverse (List.map id (Multiset.powersetAux' l). …
        -/
      · simp
        /-
          🎉 no goals
        -/
      /-
        α : Type u_1
        a : α
        s : Multiset α
        l : List α
        ⊢ Eq (Multiset.powersetAux' l).length (List.map (Multiset.cons a) (Multiset.po …
      -/
    · simp
      /-
        🎉 no goals
      -/


theorem antidiagonal_eq_map_powerset [DecidableEq α] (s : Multiset α) :
    s.antidiagonal = s.powerset.map fun t ↦ (s - t, t) := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s : Multiset α
    ⊢ Eq s.antidiagonal (Multiset.map (fun t => { fst := HSub.hSub s t, snd := t } …
  -/
  induction' s using Multiset.induction_on with a s hs
    /-
      case empty
      α : Type u_1
      inst✝ : DecidableEq α
      ⊢ Eq (Multiset.antidiagonal 0) (Multiset.map (fun t => { fst := HSub.hSub 0 t, …
    -/
  · simp only [antidiagonal_zero, powerset_zero, Multiset.zero_sub, map_singleton]
    /-
      🎉 no goals
    -/
  · simp_rw [antidiagonal_cons, powerset_cons, map_add, hs, map_map, Function.comp, Prod.map_apply,
      id, sub_cons, erase_cons_head]
    /-
      case cons
      α : Type u_1
      inst✝ : DecidableEq α
      a : α
      s : Multiset α
      hs : Eq s.antidiagonal (Multiset.map (fun t => { fst := HSub.hSub s t, snd :=  …
      ⊢ Eq (HAdd.hAdd (Multiset.map (fun x => { fst := HSub.hSub s x, snd := Multise …
    -/
    rw [add_comm]
    /-
      case cons
      α : Type u_1
      inst✝ : DecidableEq α
      a : α
      s : Multiset α
      hs : Eq s.antidiagonal (Multiset.map (fun t => { fst := HSub.hSub s t, snd :=  …
      ⊢ Eq (HAdd.hAdd (Multiset.map (fun x => { fst := Multiset.cons a (HSub.hSub s  …
    -/
    congr 1
    /-
      case cons.e_a
      α : Type u_1
      inst✝ : DecidableEq α
      a : α
      s : Multiset α
      hs : Eq s.antidiagonal (Multiset.map (fun t => { fst := HSub.hSub s t, snd :=  …
      ⊢ Eq (Multiset.map (fun x => { fst := Multiset.cons a (HSub.hSub s x), snd :=  …
    -/
    refine Multiset.map_congr rfl fun x hx ↦ ?_
    /-
      case cons.e_a
      α : Type u_1
      inst✝ : DecidableEq α
      a : α
      s : Multiset α
      hs : Eq s.antidiagonal (Multiset.map (fun t => { fst := HSub.hSub s t, snd :=  …
      x : Multiset α
      hx : Membership.mem s.powerset x
      ⊢ Eq { fst := Multiset.cons a (HSub.hSub s x), snd := x } { fst := HSub.hSub ( …
    -/
    rw [cons_sub_of_le _ (mem_powerset.mp hx)]
    /-
      🎉 no goals
    -/


@[simp]
theorem card_antidiagonal (s : Multiset α) : card (antidiagonal s) = 2 ^ card s := by
  /-
    α : Type u_1
    s : Multiset α
    ⊢ Eq s.antidiagonal.card (HPow.hPow 2 s.card)
  -/
  have := card_powerset s
  /-
    α : Type u_1
    s : Multiset α
    this : Eq s.powerset.card (HPow.hPow 2 s.card)
    ⊢ Eq s.antidiagonal.card (HPow.hPow 2 s.card)
  -/
  rwa [← antidiagonal_map_fst, card_map] at this
  /-
    🎉 no goals
  -/


