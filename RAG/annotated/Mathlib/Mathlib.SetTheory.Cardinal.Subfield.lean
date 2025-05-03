private abbrev Operands : Fin 6 ⊕ s → Type
  | .inl 0 => Bool -- add
  | .inl 1 => Bool -- mul
  | .inl 2 => Unit -- neg
  | .inl 3 => Unit -- inv
  | .inl 4 => Empty -- zero
  | .inl 5 => Empty -- one
  | .inr _ => Empty -- s


private def operate : (Σ n, Operands s n → closure s) → closure s
  | ⟨.inl 0, f⟩ => f false + f true
  | ⟨.inl 1, f⟩ => f false * f true
  | ⟨.inl 2, f⟩ => - f ()
  | ⟨.inl 3, f⟩ => (f ())⁻¹
  | ⟨.inl 4, _⟩ => 0
  | ⟨.inl 5, _⟩ => 1
  | ⟨.inr a, _⟩ => ⟨a, subset_closure a.prop⟩


private def rangeOfWType : Subfield (closure s) where
  carrier := Set.range (WType.elim _ <| operate s)
                 /-
                   α : Type u
                   s : Set α
                   inst✝ : DivisionRing α
                   ⊢ ∀ {a b : Subtype fun x => Membership.mem (Subfield.closure s) x}, Membership …
                 -/
                 /-
                   α : Type u
                   s : Set α
                   inst✝ : DivisionRing α
                   ⊢ ∀ {a b : Subtype fun x => Membership.mem (Subfield.closure s) x}, Membership …
                 -/
  add_mem' := by rintro _ _ ⟨x, rfl⟩ ⟨y, rfl⟩; exact ⟨WType.mk (.inl 0) (Bool.rec x y), by rfl⟩
                                               /-
                                                 🎉 no goals
                                               -/
                                               /-
                                                 🎉 no goals
                                               -/
  mul_mem' := by rintro _ _ ⟨x, rfl⟩ ⟨y, rfl⟩; exact ⟨WType.mk (.inl 1) (Bool.rec x y), by rfl⟩
                 /-
                   α : Type u
                   s : Set α
                   inst✝ : DivisionRing α
                   ⊢ ∀ {x : Subtype fun x => Membership.mem (Subfield.closure s) x}, Membership.m …
                 -/
  neg_mem' := by rintro _ ⟨x, rfl⟩; exact ⟨WType.mk (.inl 2) fun _ ↦ x, rfl⟩
                                    /-
                                      🎉 no goals
                                    -/
                 /-
                   α : Type u
                   s : Set α
                   inst✝ : DivisionRing α
                   ⊢ ∀ (x : Subtype fun x => Membership.mem (Subfield.closure s) x), Membership.m …
                 -/
  inv_mem' := by rintro _ ⟨x, rfl⟩; exact ⟨WType.mk (.inl 3) fun _ ↦ x, rfl⟩
                                    /-
                                      🎉 no goals
                                    -/
  zero_mem' := ⟨WType.mk (.inl 4) Empty.rec, rfl⟩
  one_mem' := ⟨WType.mk (.inl 5) Empty.rec, rfl⟩


private lemma rangeOfWType_eq_top : rangeOfWType s = ⊤ := top_le_iff.mp fun a _ ↦ by
  /-
    α : Type u
    s : Set α
    inst✝ : DivisionRing α
    a : Subtype fun x => Membership.mem (Subfield.closure s) x
    x✝ : Membership.mem Top.top a
    ⊢ Membership.mem (Subfield.rangeOfWType s) a
  -/
  rw [← SetLike.mem_coe, ← Subtype.val_injective.mem_set_image]
  /-
    α : Type u
    s : Set α
    inst✝ : DivisionRing α
    a : Subtype fun x => Membership.mem (Subfield.closure s) x
    x✝ : Membership.mem Top.top a
    ⊢ Membership.mem (Set.image Subtype.val ↑(Subfield.rangeOfWType s)) ↑a
  -/
  change ↑a ∈ map (closure s).subtype _
  /-
    α : Type u
    s : Set α
    inst✝ : DivisionRing α
    a : Subtype fun x => Membership.mem (Subfield.closure s) x
    x✝ : Membership.mem Top.top a
    ⊢ Membership.mem (Subfield.map (Subfield.closure s).subtype (Subfield.rangeOfW …
  -/
  refine closure_le.mpr (fun a ha ↦ ?_) a.prop
  /-
    α : Type u
    s : Set α
    inst✝ : DivisionRing α
    a✝ : Subtype fun x => Membership.mem (Subfield.closure s) x
    x✝ : Membership.mem Top.top a✝
    a : α
    ha : Membership.mem s a
    ⊢ Membership.mem (↑(Subfield.map (Subfield.closure s).subtype (Subfield.rangeO …
  -/
  exact ⟨⟨a, subset_closure ha⟩, ⟨WType.mk (.inr ⟨a, ha⟩) Empty.rec, rfl⟩, rfl⟩
  /-
    🎉 no goals
  -/


private lemma surjective_ofWType : Function.Surjective (WType.elim _ <| operate s) := by
  /-
    α : Type u
    s : Set α
    inst✝ : DivisionRing α
    ⊢ Function.Surjective (WType.elim (Subtype fun x => Membership.mem (Subfield.c …
  -/
  rw [← Set.range_eq_univ]
  /-
    α : Type u
    s : Set α
    inst✝ : DivisionRing α
    ⊢ Eq (Set.range (WType.elim (Subtype fun x => Membership.mem (Subfield.closure …
  -/
  exact SetLike.coe_set_eq.mpr (rangeOfWType_eq_top s)
  /-
    🎉 no goals
  -/


lemma cardinalMk_closure_le_max : #(closure s) ≤ max #s ℵ₀ :=
  (Cardinal.mk_le_of_surjective <| surjective_ofWType s).trans <| by
    /-
      α : Type u
      s : Set α
      inst✝ : DivisionRing α
      ⊢ LE.le (Cardinal.mk (WType (Subfield.Operands s))) (Max.max (Cardinal.mk ↑s)  …
    -/
    convert WType.cardinalMk_le_max_aleph0_of_finite' using 1
      /-
        case h.e'_4
        α : Type u
        s : Set α
        inst✝ : DivisionRing α
        ⊢ Eq (Max.max (Cardinal.mk ↑s) Cardinal.aleph0) (Max.max (Cardinal.lift.{0, u} …
      -/
    · rw [lift_uzero, mk_sum, lift_uzero]
      /-
        case h.e'_4
        α : Type u
        s : Set α
        inst✝ : DivisionRing α
        ⊢ Eq (Max.max (Cardinal.mk ↑s) Cardinal.aleph0) (Max.max (HAdd.hAdd (Cardinal. …
      -/
      have : lift.{u,0} #(Fin 6) < ℵ₀ := lift_lt_aleph0.mpr (lt_aleph0_of_finite _)
      /-
        case h.e'_4
        α : Type u
        s : Set α
        inst✝ : DivisionRing α
        this : LT.lt (Cardinal.lift.{u, 0} (Cardinal.mk (Fin 6))) Cardinal.aleph0
        ⊢ Eq (Max.max (Cardinal.mk ↑s) Cardinal.aleph0) (Max.max (HAdd.hAdd (Cardinal. …
      -/
      obtain h|h := lt_or_le #s ℵ₀
        /-
          case h.e'_4.inl
          α : Type u
          s : Set α
          inst✝ : DivisionRing α
          this : LT.lt (Cardinal.lift.{u, 0} (Cardinal.mk (Fin 6))) Cardinal.aleph0
          h : LT.lt (Cardinal.mk ↑s) Cardinal.aleph0
          ⊢ Eq (Max.max (Cardinal.mk ↑s) Cardinal.aleph0) (Max.max (HAdd.hAdd (Cardinal. …
        -/
      · rw [max_eq_right h.le, max_eq_right]
        /-
          case h.e'_4.inl
          α : Type u
          s : Set α
          inst✝ : DivisionRing α
          this : LT.lt (Cardinal.lift.{u, 0} (Cardinal.mk (Fin 6))) Cardinal.aleph0
          h : LT.lt (Cardinal.mk ↑s) Cardinal.aleph0
          ⊢ LE.le (HAdd.hAdd (Cardinal.lift.{u, 0} (Cardinal.mk (Fin 6))) (Cardinal.mk ↑ …
        -/
        exact (add_lt_aleph0 this h).le
        /-
          🎉 no goals
        -/
        /-
          case h.e'_4.inr
          α : Type u
          s : Set α
          inst✝ : DivisionRing α
          this : LT.lt (Cardinal.lift.{u, 0} (Cardinal.mk (Fin 6))) Cardinal.aleph0
          h : LE.le Cardinal.aleph0 (Cardinal.mk ↑s)
          ⊢ Eq (Max.max (Cardinal.mk ↑s) Cardinal.aleph0) (Max.max (HAdd.hAdd (Cardinal. …
        -/
      · rw [max_eq_left h, add_eq_right h (this.le.trans h), max_eq_left h]
        /-
          🎉 no goals
        -/
    /-
      case convert_3
      α : Type u
      s : Set α
      inst✝ : DivisionRing α
      ⊢ ∀ (a : Sum (Fin 6) ↑s), Finite (Subfield.Operands s a)
    -/
    rintro (n|_)
      /-
        case convert_3.inl
        α : Type u
        s : Set α
        inst✝ : DivisionRing α
        n : Fin 6
        ⊢ Finite (Subfield.Operands s (Sum.inl n))
      -/
                                           /-
                                             🎉 no goals
                                           -/
                                           /-
                                             🎉 no goals
                                           -/
                                           /-
                                             🎉 no goals
                                           -/
                                           /-
                                             🎉 no goals
                                           -/
                                           /-
                                             🎉 no goals
                                           -/
    · fin_cases n <;> (dsimp only [id_eq]; infer_instance)
                                           /-
                                             🎉 no goals
                                           -/
    /-
      case convert_3.inr
      α : Type u
      s : Set α
      inst✝ : DivisionRing α
      val✝ : ↑s
      ⊢ Finite (Subfield.Operands s (Sum.inr val✝))
    -/
    infer_instance
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-11-10")] alias cardinal_mk_closure_le_max := cardinalMk_closure_le_max


lemma cardinalMk_closure [Infinite s] : #(closure s) = #s :=
  ((cardinalMk_closure_le_max s).trans_eq <| max_eq_left <| aleph0_le_mk s).antisymm
    (mk_le_mk_of_subset subset_closure)


@[deprecated (since := "2024-11-10")] alias cardinal_mk_closure := cardinalMk_closure


