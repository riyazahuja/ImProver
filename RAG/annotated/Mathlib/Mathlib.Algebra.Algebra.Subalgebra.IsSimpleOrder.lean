theorem Subalgebra.isSimpleOrder_of_finrank_prime (F A) [Field F] [Ring A] [IsDomain A]
    [Algebra F A] (hp : (finrank F A).Prime) : IsSimpleOrder (Subalgebra F A) :=
  { toNontrivial :=
      ⟨⟨⊥, ⊤, fun he =>
          Nat.not_prime_one ((Subalgebra.bot_eq_top_iff_finrank_eq_one.1 he).subst hp)⟩⟩
    eq_bot_or_eq_top := fun K => by
      /-
        F : Type u_1
        A : Type u_2
        inst✝³ : Field F
        inst✝² : Ring A
        inst✝¹ : IsDomain A
        inst✝ : Algebra F A
        hp : Nat.Prime (Module.finrank F A)
        K : Subalgebra F A
        ⊢ Or (Eq K Bot.bot) (Eq K Top.top)
      -/
      haveI : FiniteDimensional _ _ := .of_finrank_pos hp.pos
      /-
        F : Type u_1
        A : Type u_2
        inst✝³ : Field F
        inst✝² : Ring A
        inst✝¹ : IsDomain A
        inst✝ : Algebra F A
        hp : Nat.Prime (Module.finrank F A)
        K : Subalgebra F A
        this : FiniteDimensional F A
        ⊢ Or (Eq K Bot.bot) (Eq K Top.top)
      -/
      letI := divisionRingOfFiniteDimensional F K
      /-
        F : Type u_1
        A : Type u_2
        inst✝³ : Field F
        inst✝² : Ring A
        inst✝¹ : IsDomain A
        inst✝ : Algebra F A
        hp : Nat.Prime (Module.finrank F A)
        K : Subalgebra F A
        this✝ : FiniteDimensional F A
        this : DivisionRing (Subtype fun x => Membership.mem K x) := divisionRingOfFin …
        ⊢ Or (Eq K Bot.bot) (Eq K Top.top)
      -/
      refine (hp.eq_one_or_self_of_dvd _ ⟨_, (finrank_mul_finrank F K A).symm⟩).imp ?_ fun h => ?_
        /-
          case refine_1
          F : Type u_1
          A : Type u_2
          inst✝³ : Field F
          inst✝² : Ring A
          inst✝¹ : IsDomain A
          inst✝ : Algebra F A
          hp : Nat.Prime (Module.finrank F A)
          K : Subalgebra F A
          this✝ : FiniteDimensional F A
          this : DivisionRing (Subtype fun x => Membership.mem K x) := divisionRingOfFin …
          ⊢ Eq (Module.finrank F (Subtype fun x => Membership.mem K x)) 1 → Eq K Bot.bot
        -/
      · exact fun h' => Subalgebra.eq_bot_of_finrank_one h'
        /-
          🎉 no goals
        -/
      · exact
          Algebra.toSubmodule_eq_top.1 (eq_top_of_finrank_eq <| K.finrank_toSubmodule.trans h) }
-- TODO: `IntermediateField` version


@[deprecated (since := "2024-08-11")]
alias FiniteDimensional.Subalgebra.is_simple_order_of_finrank_prime :=
  Subalgebra.isSimpleOrder_of_finrank_prime

