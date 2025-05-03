/-- A relation on the free monoid on alphabet `Σ i, M i`,
relating `⟨i, 1⟩` with `1` and `⟨i, x⟩ * ⟨i, y⟩` with `⟨i, x * y⟩`. -/
inductive Monoid.CoprodI.Rel : FreeMonoid (Σi, M i) → FreeMonoid (Σi, M i) → Prop
  | of_one (i : ι) : Monoid.CoprodI.Rel (FreeMonoid.of ⟨i, 1⟩) 1
  | of_mul {i : ι} (x y : M i) :
    Monoid.CoprodI.Rel (FreeMonoid.of ⟨i, x⟩ * FreeMonoid.of ⟨i, y⟩) (FreeMonoid.of ⟨i, x * y⟩)


/-- The free product (categorical coproduct) of an indexed family of monoids. -/
def Monoid.CoprodI : Type _ := (conGen (Monoid.CoprodI.Rel M)).Quotient

-- Porting note: could not de derived

instance : Monoid (Monoid.CoprodI M) := by
  /-
    ι : Type u_1
    M : ι → Type u_2
    inst✝ : (i : ι) → Monoid (M i)
    ⊢ Monoid (Monoid.CoprodI M)
  -/
  delta Monoid.CoprodI; infer_instance
                        /-
                          🎉 no goals
                        -/


instance : Inhabited (Monoid.CoprodI M) :=
  ⟨1⟩


/-- The type of reduced words. A reduced word cannot contain a letter `1`, and no two adjacent
letters can come from the same summand. -/
@[ext]
structure Word where
  /-- A `Word` is a `List (Σ i, M i)`, such that `1` is not in the list, and no
  two adjacent letters are from the same summand -/
  toList : List (Σi, M i)
  /-- A reduced word does not contain `1` -/
  ne_one : ∀ l ∈ toList, Sigma.snd l ≠ 1
  /-- Adjacent letters are not from the same summand. -/
  chain_ne : toList.Chain' fun l l' => Sigma.fst l ≠ Sigma.fst l'


/-- The inclusion of a summand into the free product. -/
def of {i : ι} : M i →* CoprodI M where
  toFun x := Con.mk' _ (FreeMonoid.of <| Sigma.mk i x)
  map_one' := (Con.eq _).mpr (ConGen.Rel.of _ _ (CoprodI.Rel.of_one i))
  map_mul' x y := Eq.symm <| (Con.eq _).mpr (ConGen.Rel.of _ _ (CoprodI.Rel.of_mul x y))


theorem of_apply {i} (m : M i) : of m = Con.mk' _ (FreeMonoid.of <| Sigma.mk i m) :=
  rfl


/-- See note [partially-applied ext lemmas]. -/
-- Porting note: higher `ext` priority
@[ext 1100]
theorem ext_hom (f g : CoprodI M →* N) (h : ∀ i, f.comp (of : M i →* _) = g.comp of) : f = g :=
  (MonoidHom.cancel_right Con.mk'_surjective).mp <|
    FreeMonoid.hom_eq fun ⟨i, x⟩ => by
      -- This used to be `rw`, but we need `erw` after https://github.com/leanprover/lean4/pull/2644
      erw [MonoidHom.comp_apply, MonoidHom.comp_apply, ← of_apply, ← MonoidHom.comp_apply, ←
                                  /-
                                    ι : Type u_1
                                    M : ι → Type u_2
                                    inst✝¹ : (i : ι) → Monoid (M i)
                                    N : Type u_3
                                    inst✝ : Monoid N
                                    f g : MonoidHom (Monoid.CoprodI M) N
                                    h : ∀ (i : ι), Eq (f.comp Monoid.CoprodI.of) (g.comp Monoid.CoprodI.of)
                                    x✝ : Sigma fun i => M i
                                    i : ι
                                    x : M i
                                    ⊢ Eq ((g.comp Monoid.CoprodI.of) x) ((g.comp (conGen (Monoid.CoprodI.Rel M)).m …
                                  -/
        MonoidHom.comp_apply, h]; rfl
                                  /-
                                    🎉 no goals
                                  -/


/-- A map out of the free product corresponds to a family of maps out of the summands. This is the
universal property of the free product, characterizing it as a categorical coproduct. -/
@[simps symm_apply]
def lift : (∀ i, M i →* N) ≃ (CoprodI M →* N) where
  toFun fi :=
    Con.lift _ (FreeMonoid.lift fun p : Σi, M i => fi p.fst p.snd) <|
      Con.conGen_le <| by
        /-
          ι : Type u_1
          M : ι → Type u_2
          inst✝¹ : (i : ι) → Monoid (M i)
          N : Type u_3
          inst✝ : Monoid N
          fi : (i : ι) → MonoidHom (M i) N
          ⊢ ∀ (x y : FreeMonoid (Sigma fun i => M i)), Monoid.CoprodI.Rel M x y → (Con.k …
        -/
        simp_rw [Con.ker_rel]
        /-
          ι : Type u_1
          M : ι → Type u_2
          inst✝¹ : (i : ι) → Monoid (M i)
          N : Type u_3
          inst✝ : Monoid N
          fi : (i : ι) → MonoidHom (M i) N
          ⊢ ∀ (x y : FreeMonoid (Sigma fun i => M i)), Monoid.CoprodI.Rel M x y → Eq ((F …
        -/
        rintro _ _ (i | ⟨x, y⟩)
          /-
            case of_one
            ι : Type u_1
            M : ι → Type u_2
            inst✝¹ : (i : ι) → Monoid (M i)
            N : Type u_3
            inst✝ : Monoid N
            fi : (i : ι) → MonoidHom (M i) N
            i : ι
            ⊢ Eq ((FreeMonoid.lift fun p => (fi p.fst) p.snd) (FreeMonoid.of ⟨i, 1⟩)) ((Fr …
          -/
        · change FreeMonoid.lift _ (FreeMonoid.of _) = FreeMonoid.lift _ 1
          /-
            case of_one
            ι : Type u_1
            M : ι → Type u_2
            inst✝¹ : (i : ι) → Monoid (M i)
            N : Type u_3
            inst✝ : Monoid N
            fi : (i : ι) → MonoidHom (M i) N
            i : ι
            ⊢ Eq ((FreeMonoid.lift fun p => (fi p.fst) p.snd) (FreeMonoid.of ⟨i, 1⟩)) ((Fr …
          -/
          simp only [MonoidHom.map_one, FreeMonoid.lift_eval_of]
          /-
            🎉 no goals
          -/
        · change
            FreeMonoid.lift _ (FreeMonoid.of _ * FreeMonoid.of _) =
              FreeMonoid.lift _ (FreeMonoid.of _)
          /-
            case of_mul
            ι : Type u_1
            M : ι → Type u_2
            inst✝¹ : (i : ι) → Monoid (M i)
            N : Type u_3
            inst✝ : Monoid N
            fi : (i : ι) → MonoidHom (M i) N
            i✝ : ι
            x y : M i✝
            ⊢ Eq ((FreeMonoid.lift fun p => (fi p.fst) p.snd) (HMul.hMul (FreeMonoid.of ⟨i …
          -/
          simp only [MonoidHom.map_mul, FreeMonoid.lift_eval_of]
          /-
            🎉 no goals
          -/
  invFun f _ := f.comp of
  left_inv := by
    /-
      ι : Type u_1
      M : ι → Type u_2
      inst✝¹ : (i : ι) → Monoid (M i)
      N : Type u_3
      inst✝ : Monoid N
      ⊢ Function.LeftInverse (fun f x => f.comp Monoid.CoprodI.of) fun fi => (conGen …
    -/
    intro fi
    /-
      ι : Type u_1
      M : ι → Type u_2
      inst✝¹ : (i : ι) → Monoid (M i)
      N : Type u_3
      inst✝ : Monoid N
      fi : (i : ι) → MonoidHom (M i) N
      ⊢ Eq ((fun f x => f.comp Monoid.CoprodI.of) ((fun fi => (conGen (Monoid.Coprod …
    -/
    ext i x
    -- This used to be `rw`, but we need `erw` after https://github.com/leanprover/lean4/pull/2644
    /-
      case h.h
      ι : Type u_1
      M : ι → Type u_2
      inst✝¹ : (i : ι) → Monoid (M i)
      N : Type u_3
      inst✝ : Monoid N
      fi : (i : ι) → MonoidHom (M i) N
      i : ι
      x : M i
      ⊢ Eq (((fun f x => f.comp Monoid.CoprodI.of) ((fun fi => (conGen (Monoid.Copro …
    -/
    erw [MonoidHom.comp_apply, of_apply, Con.lift_mk', FreeMonoid.lift_eval_of]
    /-
      🎉 no goals
    -/
  right_inv := by
    /-
      ι : Type u_1
      M : ι → Type u_2
      inst✝¹ : (i : ι) → Monoid (M i)
      N : Type u_3
      inst✝ : Monoid N
      ⊢ Function.RightInverse (fun f x => f.comp Monoid.CoprodI.of) fun fi => (conGe …
    -/
    intro f
    /-
      ι : Type u_1
      M : ι → Type u_2
      inst✝¹ : (i : ι) → Monoid (M i)
      N : Type u_3
      inst✝ : Monoid N
      f : MonoidHom (Monoid.CoprodI M) N
      ⊢ Eq ((fun fi => (conGen (Monoid.CoprodI.Rel M)).lift (FreeMonoid.lift fun p = …
    -/
    ext i x
    /-
      case h.h
      ι : Type u_1
      M : ι → Type u_2
      inst✝¹ : (i : ι) → Monoid (M i)
      N : Type u_3
      inst✝ : Monoid N
      f : MonoidHom (Monoid.CoprodI M) N
      i : ι
      x : M i
      ⊢ Eq ((((fun fi => (conGen (Monoid.CoprodI.Rel M)).lift (FreeMonoid.lift fun p …
    -/
    rfl
    /-
      🎉 no goals
    -/


@[simp]
theorem lift_comp_of {N} [Monoid N] (fi : ∀ i, M i →* N) i : (lift fi).comp of = fi i :=
  congr_fun (lift.symm_apply_apply fi) i


@[simp]
theorem lift_of {N} [Monoid N] (fi : ∀ i, M i →* N) {i} (m : M i) : lift fi (of m) = fi i m :=
  DFunLike.congr_fun (lift_comp_of ..) m


@[simp]
theorem lift_comp_of' {N} [Monoid N] (f : CoprodI M →* N) :
    lift (fun i ↦ f.comp (of (i := i))) = f :=
  lift.apply_symm_apply f


@[simp]
theorem lift_of' : lift (fun i ↦ (of : M i →* CoprodI M)) = .id (CoprodI M) :=
  lift_comp_of' (.id _)


theorem of_leftInverse [DecidableEq ι] (i : ι) :
    Function.LeftInverse (lift <| Pi.mulSingle i (MonoidHom.id (M i))) of := fun x => by
  /-
    ι : Type u_1
    M : ι → Type u_2
    inst✝¹ : (i : ι) → Monoid (M i)
    inst✝ : DecidableEq ι
    i : ι
    x : M i
    ⊢ Eq ((Monoid.CoprodI.lift (Pi.mulSingle i (MonoidHom.id (M i)))) (Monoid.Copr …
  -/
  simp only [lift_of, Pi.mulSingle_eq_same, MonoidHom.id_apply]
  /-
    🎉 no goals
  -/


theorem of_injective (i : ι) : Function.Injective (of : M i →* _) := by
  /-
    ι : Type u_1
    M : ι → Type u_2
    inst✝ : (i : ι) → Monoid (M i)
    i : ι
    ⊢ Function.Injective ⇑Monoid.CoprodI.of
  -/
  classical exact (of_leftInverse i).injective
  /-
    🎉 no goals
  -/


theorem mrange_eq_iSup {N} [Monoid N] (f : ∀ i, M i →* N) :
    MonoidHom.mrange (lift f) = ⨆ i, MonoidHom.mrange (f i) := by
  rw [lift, Equiv.coe_fn_mk, Con.lift_range, FreeMonoid.mrange_lift,
    range_sigma_eq_iUnion_range, Submonoid.closure_iUnion]
  /-
    ι : Type u_1
    M : ι → Type u_2
    inst✝¹ : (i : ι) → Monoid (M i)
    N : Type u_4
    inst✝ : Monoid N
    f : (i : ι) → MonoidHom (M i) N
    ⊢ Eq (iSup fun i => Submonoid.closure (Set.range fun b => (f ⟨i, b⟩.fst) ⟨i, b …
  -/
  simp only [MonoidHom.mclosure_range]
  /-
    🎉 no goals
  -/


theorem lift_mrange_le {N} [Monoid N] (f : ∀ i, M i →* N) {s : Submonoid N} :
    MonoidHom.mrange (lift f) ≤ s ↔ ∀ i, MonoidHom.mrange (f i) ≤ s := by
  /-
    ι : Type u_1
    M : ι → Type u_2
    inst✝¹ : (i : ι) → Monoid (M i)
    N : Type u_4
    inst✝ : Monoid N
    f : (i : ι) → MonoidHom (M i) N
    s : Submonoid N
    ⊢ Iff (LE.le (MonoidHom.mrange (Monoid.CoprodI.lift f)) s) (∀ (i : ι), LE.le ( …
  -/
  simp [mrange_eq_iSup]
  /-
    🎉 no goals
  -/


@[simp]
theorem iSup_mrange_of : ⨆ i, MonoidHom.mrange (of : M i →* CoprodI M) = ⊤ := by
  /-
    ι : Type u_1
    M : ι → Type u_2
    inst✝ : (i : ι) → Monoid (M i)
    ⊢ Eq (iSup fun i => MonoidHom.mrange Monoid.CoprodI.of) Top.top
  -/
  simp [← mrange_eq_iSup]
  /-
    🎉 no goals
  -/


@[simp]
theorem mclosure_iUnion_range_of :
    Submonoid.closure (⋃ i, Set.range (of : M i →* CoprodI M)) = ⊤ := by
  /-
    ι : Type u_1
    M : ι → Type u_2
    inst✝ : (i : ι) → Monoid (M i)
    ⊢ Eq (Submonoid.closure (Set.iUnion fun i => Set.range ⇑Monoid.CoprodI.of)) To …
  -/
  simp [Submonoid.closure_iUnion]
  /-
    🎉 no goals
  -/


@[elab_as_elim]
theorem induction_left {C : CoprodI M → Prop} (m : CoprodI M) (one : C 1)
    (mul : ∀ {i} (m : M i) x, C x → C (of m * x)) : C m := by
  induction m using Submonoid.induction_of_closure_eq_top_left mclosure_iUnion_range_of with
  | one => exact one
  | mul x hx y ihy =>
    obtain ⟨i, m, rfl⟩ : ∃ (i : ι) (m : M i), of m = x := by simpa using hx
    exact mul m y ihy


@[elab_as_elim]
theorem induction_on {C : CoprodI M → Prop} (m : CoprodI M) (h_one : C 1)
    (h_of : ∀ (i) (m : M i), C (of m)) (h_mul : ∀ x y, C x → C y → C (x * y)) : C m := by
  induction m using CoprodI.induction_left with
  | one => exact h_one
  | mul m x hx => exact h_mul _ _ (h_of _ _) hx


instance : Inv (CoprodI G) where
  inv :=
    MulOpposite.unop ∘ lift fun i => (of : G i →* _).op.comp (MulEquiv.inv' (G i)).toMonoidHom


theorem inv_def (x : CoprodI G) :
    x⁻¹ =
      MulOpposite.unop
        (lift (fun i => (of : G i →* _).op.comp (MulEquiv.inv' (G i)).toMonoidHom) x) :=
  rfl


instance : Group (CoprodI G) :=
  { inv_mul_cancel := by
      /-
        ι : Type u_1
        M : ι → Type u_2
        inst✝² : (i : ι) → Monoid (M i)
        N : Type u_3
        inst✝¹ : Monoid N
        G : ι → Type u_4
        inst✝ : (i : ι) → Group (G i)
        ⊢ ∀ (a : Monoid.CoprodI G), Eq (HMul.hMul (Inv.inv a) a) 1
      -/
      intro m
      /-
        ι : Type u_1
        M : ι → Type u_2
        inst✝² : (i : ι) → Monoid (M i)
        N : Type u_3
        inst✝¹ : Monoid N
        G : ι → Type u_4
        inst✝ : (i : ι) → Group (G i)
        m : Monoid.CoprodI G
        ⊢ Eq (HMul.hMul (Inv.inv m) m) 1
      -/
      rw [inv_def]
      induction m using CoprodI.induction_on with
      | h_one => rw [MonoidHom.map_one, MulOpposite.unop_one, one_mul]
      | h_of m ih =>
        change of _⁻¹ * of _ = 1
        rw [← of.map_mul, inv_mul_cancel, of.map_one]
      | h_mul x y ihx ihy =>
        rw [MonoidHom.map_mul, MulOpposite.unop_mul, mul_assoc, ← mul_assoc _ x y, ihx, one_mul,
          ihy] }


theorem lift_range_le {N} [Group N] (f : ∀ i, G i →* N) {s : Subgroup N}
    (h : ∀ i, (f i).range ≤ s) : (lift f).range ≤ s := by
  /-
    ι : Type u_1
    G : ι → Type u_4
    inst✝¹ : (i : ι) → Group (G i)
    N : Type u_5
    inst✝ : Group N
    f : (i : ι) → MonoidHom (G i) N
    s : Subgroup N
    h : ∀ (i : ι), LE.le (f i).range s
    ⊢ LE.le (Monoid.CoprodI.lift f).range s
  -/
  rintro _ ⟨x, rfl⟩
  /-
    case intro
    ι : Type u_1
    G : ι → Type u_4
    inst✝¹ : (i : ι) → Group (G i)
    N : Type u_5
    inst✝ : Group N
    f : (i : ι) → MonoidHom (G i) N
    s : Subgroup N
    h : ∀ (i : ι), LE.le (f i).range s
    x : Monoid.CoprodI G
    ⊢ Membership.mem s ((Monoid.CoprodI.lift f) x)
  -/
  induction' x using CoprodI.induction_on with i x x y hx hy
    /-
      case intro.h_one
      ι : Type u_1
      G : ι → Type u_4
      inst✝¹ : (i : ι) → Group (G i)
      N : Type u_5
      inst✝ : Group N
      f : (i : ι) → MonoidHom (G i) N
      s : Subgroup N
      h : ∀ (i : ι), LE.le (f i).range s
      ⊢ Membership.mem s ((Monoid.CoprodI.lift f) 1)
    -/
  · exact s.one_mem
    /-
      🎉 no goals
    -/
    /-
      case intro.h_of
      ι : Type u_1
      G : ι → Type u_4
      inst✝¹ : (i : ι) → Group (G i)
      N : Type u_5
      inst✝ : Group N
      f : (i : ι) → MonoidHom (G i) N
      s : Subgroup N
      h : ∀ (i : ι), LE.le (f i).range s
      i : ι
      x : G i
      ⊢ Membership.mem s ((Monoid.CoprodI.lift f) (Monoid.CoprodI.of x))
    -/
  · simp only [lift_of, SetLike.mem_coe]
    /-
      case intro.h_of
      ι : Type u_1
      G : ι → Type u_4
      inst✝¹ : (i : ι) → Group (G i)
      N : Type u_5
      inst✝ : Group N
      f : (i : ι) → MonoidHom (G i) N
      s : Subgroup N
      h : ∀ (i : ι), LE.le (f i).range s
      i : ι
      x : G i
      ⊢ Membership.mem s ((f i) x)
    -/
    exact h i (Set.mem_range_self x)
    /-
      🎉 no goals
    -/
    /-
      case intro.h_mul
      ι : Type u_1
      G : ι → Type u_4
      inst✝¹ : (i : ι) → Group (G i)
      N : Type u_5
      inst✝ : Group N
      f : (i : ι) → MonoidHom (G i) N
      s : Subgroup N
      h : ∀ (i : ι), LE.le (f i).range s
      x y : Monoid.CoprodI G
      hx : Membership.mem s ((Monoid.CoprodI.lift f) x)
      hy : Membership.mem s ((Monoid.CoprodI.lift f) y)
      ⊢ Membership.mem s ((Monoid.CoprodI.lift f) (HMul.hMul x y))
    -/
  · simp only [map_mul, SetLike.mem_coe]
    /-
      case intro.h_mul
      ι : Type u_1
      G : ι → Type u_4
      inst✝¹ : (i : ι) → Group (G i)
      N : Type u_5
      inst✝ : Group N
      f : (i : ι) → MonoidHom (G i) N
      s : Subgroup N
      h : ∀ (i : ι), LE.le (f i).range s
      x y : Monoid.CoprodI G
      hx : Membership.mem s ((Monoid.CoprodI.lift f) x)
      hy : Membership.mem s ((Monoid.CoprodI.lift f) y)
      ⊢ Membership.mem s (HMul.hMul ((Monoid.CoprodI.lift f) x) ((Monoid.CoprodI.lif …
    -/
    exact s.mul_mem hx hy
    /-
      🎉 no goals
    -/


theorem range_eq_iSup {N} [Group N] (f : ∀ i, G i →* N) : (lift f).range = ⨆ i, (f i).range := by
  /-
    ι : Type u_1
    G : ι → Type u_4
    inst✝¹ : (i : ι) → Group (G i)
    N : Type u_5
    inst✝ : Group N
    f : (i : ι) → MonoidHom (G i) N
    ⊢ Eq (Monoid.CoprodI.lift f).range (iSup fun i => (f i).range)
  -/
  apply le_antisymm (lift_range_le _ f fun i => le_iSup (fun i => MonoidHom.range (f i)) i)
  /-
    ι : Type u_1
    G : ι → Type u_4
    inst✝¹ : (i : ι) → Group (G i)
    N : Type u_5
    inst✝ : Group N
    f : (i : ι) → MonoidHom (G i) N
    ⊢ LE.le (iSup fun i => (f i).range) (Monoid.CoprodI.lift f).range
  -/
  apply iSup_le _
  /-
    ι : Type u_1
    G : ι → Type u_4
    inst✝¹ : (i : ι) → Group (G i)
    N : Type u_5
    inst✝ : Group N
    f : (i : ι) → MonoidHom (G i) N
    ⊢ ∀ (i : ι), LE.le (f i).range (Monoid.CoprodI.lift f).range
  -/
  rintro i _ ⟨x, rfl⟩
  /-
    case intro
    ι : Type u_1
    G : ι → Type u_4
    inst✝¹ : (i : ι) → Group (G i)
    N : Type u_5
    inst✝ : Group N
    f : (i : ι) → MonoidHom (G i) N
    i : ι
    x : G i
    ⊢ Membership.mem (Monoid.CoprodI.lift f).range ((f i) x)
  -/
  exact ⟨of x, by simp only [lift_of]⟩
  /-
    🎉 no goals
  -/


/-- The empty reduced word. -/
@[simps]
def empty : Word M where
  toList := []
               /-
                 ι : Type u_1
                 M : ι → Type u_2
                 inst✝¹ : (i : ι) → Monoid (M i)
                 N : Type u_3
                 inst✝ : Monoid N
                 ⊢ ∀ (l : Sigma fun i => M i), Membership.mem List.nil l → Ne l.snd 1
               -/
  ne_one := by simp
               /-
                 🎉 no goals
               -/
  chain_ne := List.chain'_nil


instance : Inhabited (Word M) :=
  ⟨empty⟩


/-- A reduced word determines an element of the free product, given by multiplication. -/
def prod (w : Word M) : CoprodI M :=
  List.prod (w.toList.map fun l => of l.snd)


@[simp]
theorem prod_empty : prod (empty : Word M) = 1 :=
  rfl


/-- `fstIdx w` is `some i` if the first letter of `w` is `⟨i, m⟩` with `m : M i`. If `w` is empty
then it's `none`. -/
def fstIdx (w : Word M) : Option ι :=
  w.toList.head?.map Sigma.fst


theorem fstIdx_ne_iff {w : Word M} {i} :
    fstIdx w ≠ some i ↔ ∀ l ∈ w.toList.head?, i ≠ Sigma.fst l :=
                       /-
                         ι : Type u_1
                         M : ι → Type u_2
                         inst✝ : (i : ι) → Monoid (M i)
                         w : Monoid.CoprodI.Word M
                         i : ι
                         ⊢ Iff (Not (Ne w.fstIdx (Option.some i))) (Not (∀ (l : Sigma fun i => M i), Me …
                       -/
  not_iff_not.mp <| by simp [fstIdx]
                       /-
                         🎉 no goals
                       -/


/-- Given an index `i : ι`, `Pair M i` is the type of pairs `(head, tail)` where `head : M i` and
`tail : Word M`, subject to the constraint that first letter of `tail` can't be `⟨i, m⟩`.
By prepending `head` to `tail`, one obtains a new word. We'll show that any word can be uniquely
obtained in this way. -/
@[ext]
structure Pair (i : ι) where
  /-- An element of `M i`, the first letter of the word. -/
  head : M i
  /-- The remaining letters of the word, excluding the first letter -/
  tail : Word M
  /-- The index first letter of tail of a `Pair M i` is not equal to `i` -/
  fstIdx_ne : fstIdx tail ≠ some i


instance (i : ι) : Inhabited (Pair M i) :=
                 /-
                   ι : Type u_1
                   M : ι → Type u_2
                   inst✝¹ : (i : ι) → Monoid (M i)
                   N : Type u_3
                   inst✝ : Monoid N
                   i : ι
                   ⊢ Ne Monoid.CoprodI.Word.empty.fstIdx (Option.some i)
                 -/
  ⟨⟨1, empty, by tauto⟩⟩
                 /-
                   🎉 no goals
                 -/


/-- Construct a new `Word` without any reduction. The underlying list of
`cons m w _ _` is `⟨_, m⟩::w`  -/
@[simps]
def cons {i} (m : M i) (w : Word M) (hmw : w.fstIdx ≠ some i) (h1 : m ≠ 1) : Word M :=
  { toList := ⟨i, m⟩ :: w.toList,
    ne_one := by
      /-
        ι : Type u_1
        M : ι → Type u_2
        inst✝¹ : (i : ι) → Monoid (M i)
        N : Type u_3
        inst✝ : Monoid N
        i : ι
        m : M i
        w : Monoid.CoprodI.Word M
        hmw : Ne w.fstIdx (Option.some i)
        h1 : Ne m 1
        ⊢ ∀ (l : Sigma fun i => M i), Membership.mem (List.cons ⟨i, m⟩ w.toList) l → N …
      -/
      simp only [List.mem_cons]
      /-
        ι : Type u_1
        M : ι → Type u_2
        inst✝¹ : (i : ι) → Monoid (M i)
        N : Type u_3
        inst✝ : Monoid N
        i : ι
        m : M i
        w : Monoid.CoprodI.Word M
        hmw : Ne w.fstIdx (Option.some i)
        h1 : Ne m 1
        ⊢ ∀ (l : Sigma fun i => M i), Or (Eq l ⟨i, m⟩) (Membership.mem w.toList l) → N …
      -/
      rintro l (rfl | hl)
        /-
          case inl
          ι : Type u_1
          M : ι → Type u_2
          inst✝¹ : (i : ι) → Monoid (M i)
          N : Type u_3
          inst✝ : Monoid N
          i : ι
          m : M i
          w : Monoid.CoprodI.Word M
          hmw : Ne w.fstIdx (Option.some i)
          h1 : Ne m 1
          ⊢ Ne ⟨i, m⟩.snd 1
        -/
      · exact h1
        /-
          🎉 no goals
        -/
        /-
          case inr
          ι : Type u_1
          M : ι → Type u_2
          inst✝¹ : (i : ι) → Monoid (M i)
          N : Type u_3
          inst✝ : Monoid N
          i : ι
          m : M i
          w : Monoid.CoprodI.Word M
          hmw : Ne w.fstIdx (Option.some i)
          h1 : Ne m 1
          l : Sigma fun i => M i
          hl : Membership.mem w.toList l
          ⊢ Ne l.snd 1
        -/
      · exact w.ne_one l hl
        /-
          🎉 no goals
        -/
    chain_ne := w.chain_ne.cons' (fstIdx_ne_iff.mp hmw) }


@[simp]
theorem fstIdx_cons {i} (m : M i) (w : Word M) (hmw : w.fstIdx ≠ some i) (h1 : m ≠ 1) :
                                            /-
                                              ι : Type u_1
                                              M : ι → Type u_2
                                              inst✝ : (i : ι) → Monoid (M i)
                                              i : ι
                                              m : M i
                                              w : Monoid.CoprodI.Word M
                                              hmw : Ne w.fstIdx (Option.some i)
                                              h1 : Ne m 1
                                              ⊢ Eq (Monoid.CoprodI.Word.cons m w hmw h1).fstIdx (Option.some i)
                                            -/
    fstIdx (cons m w hmw h1) = some i := by simp [cons, fstIdx]
                                            /-
                                              🎉 no goals
                                            -/


@[simp]
theorem prod_cons (i) (m : M i) (w : Word M) (h1 : m ≠ 1) (h2 : w.fstIdx ≠ some i) :
    prod (cons m w h2 h1) = of m * prod w := by
  /-
    ι : Type u_1
    M : ι → Type u_2
    inst✝ : (i : ι) → Monoid (M i)
    i : ι
    m : M i
    w : Monoid.CoprodI.Word M
    h1 : Ne m 1
    h2 : Ne w.fstIdx (Option.some i)
    ⊢ Eq (Monoid.CoprodI.Word.cons m w h2 h1).prod (HMul.hMul (Monoid.CoprodI.of m …
  -/
  simp [cons, prod, List.map_cons, List.prod_cons]
  /-
    🎉 no goals
  -/


/-- Given a pair `(head, tail)`, we can form a word by prepending `head` to `tail`, except if `head`
is `1 : M i` then we have to just return `Word` since we need the result to be reduced. -/
def rcons {i} (p : Pair M i) : Word M :=
  if h : p.head = 1 then p.tail
  else cons p.head p.tail p.fstIdx_ne h


@[simp]
theorem prod_rcons {i} (p : Pair M i) : prod (rcons p) = of p.head * prod p.tail :=
                             /-
                               ι : Type u_1
                               M : ι → Type u_2
                               inst✝¹ : (i : ι) → Monoid (M i)
                               inst✝ : (i : ι) → DecidableEq (M i)
                               i : ι
                               p : Monoid.CoprodI.Word.Pair M i
                               hm : Eq p.head 1
                               ⊢ Eq (Monoid.CoprodI.Word.rcons p).prod (HMul.hMul (Monoid.CoprodI.of p.head)  …
                             -/
  if hm : p.head = 1 then by rw [rcons, dif_pos hm, hm, MonoidHom.map_one, one_mul]
                             /-
                               🎉 no goals
                             -/
          /-
            ι : Type u_1
            M : ι → Type u_2
            inst✝¹ : (i : ι) → Monoid (M i)
            inst✝ : (i : ι) → DecidableEq (M i)
            i : ι
            p : Monoid.CoprodI.Word.Pair M i
            hm : Not (Eq p.head 1)
            ⊢ Eq (Monoid.CoprodI.Word.rcons p).prod (HMul.hMul (Monoid.CoprodI.of p.head)  …
          -/
  else by rw [rcons, dif_neg hm, cons, prod, List.map_cons, List.prod_cons, prod]
          /-
            🎉 no goals
          -/


theorem rcons_inj {i} : Function.Injective (rcons : Pair M i → Word M) := by
  /-
    ι : Type u_1
    M : ι → Type u_2
    inst✝¹ : (i : ι) → Monoid (M i)
    inst✝ : (i : ι) → DecidableEq (M i)
    i : ι
    ⊢ Function.Injective Monoid.CoprodI.Word.rcons
  -/
  rintro ⟨m, w, h⟩ ⟨m', w', h'⟩ he
  /-
    case mk.mk
    ι : Type u_1
    M : ι → Type u_2
    inst✝¹ : (i : ι) → Monoid (M i)
    inst✝ : (i : ι) → DecidableEq (M i)
    i : ι
    m : M i
    w : Monoid.CoprodI.Word M
    h : Ne w.fstIdx (Option.some i)
    m' : M i
    w' : Monoid.CoprodI.Word M
    h' : Ne w'.fstIdx (Option.some i)
    he : Eq (Monoid.CoprodI.Word.rcons { head := m, tail := w, fstIdx_ne := h }) ( …
    ⊢ Eq { head := m, tail := w, fstIdx_ne := h } { head := m', tail := w', fstIdx …
  -/
  by_cases hm : m = 1 <;> by_cases hm' : m' = 1
    /-
      case pos
      ι : Type u_1
      M : ι → Type u_2
      inst✝¹ : (i : ι) → Monoid (M i)
      inst✝ : (i : ι) → DecidableEq (M i)
      i : ι
      m : M i
      w : Monoid.CoprodI.Word M
      h : Ne w.fstIdx (Option.some i)
      m' : M i
      w' : Monoid.CoprodI.Word M
      h' : Ne w'.fstIdx (Option.some i)
      he : Eq (Monoid.CoprodI.Word.rcons { head := m, tail := w, fstIdx_ne := h }) ( …
      hm : Eq m 1
      hm' : Eq m' 1
      ⊢ Eq { head := m, tail := w, fstIdx_ne := h } { head := m', tail := w', fstIdx …
    -/
  · simp only [rcons, dif_pos hm, dif_pos hm'] at he
    /-
      case pos
      ι : Type u_1
      M : ι → Type u_2
      inst✝¹ : (i : ι) → Monoid (M i)
      inst✝ : (i : ι) → DecidableEq (M i)
      i : ι
      m : M i
      w : Monoid.CoprodI.Word M
      h : Ne w.fstIdx (Option.some i)
      m' : M i
      w' : Monoid.CoprodI.Word M
      h' : Ne w'.fstIdx (Option.some i)
      hm : Eq m 1
      hm' : Eq m' 1
      he : Eq w w'
      ⊢ Eq { head := m, tail := w, fstIdx_ne := h } { head := m', tail := w', fstIdx …
    -/
    aesop
    /-
      🎉 no goals
    -/
    /-
      case neg
      ι : Type u_1
      M : ι → Type u_2
      inst✝¹ : (i : ι) → Monoid (M i)
      inst✝ : (i : ι) → DecidableEq (M i)
      i : ι
      m : M i
      w : Monoid.CoprodI.Word M
      h : Ne w.fstIdx (Option.some i)
      m' : M i
      w' : Monoid.CoprodI.Word M
      h' : Ne w'.fstIdx (Option.some i)
      he : Eq (Monoid.CoprodI.Word.rcons { head := m, tail := w, fstIdx_ne := h }) ( …
      hm : Eq m 1
      hm' : Not (Eq m' 1)
      ⊢ Eq { head := m, tail := w, fstIdx_ne := h } { head := m', tail := w', fstIdx …
    -/
  · exfalso
    /-
      case neg
      ι : Type u_1
      M : ι → Type u_2
      inst✝¹ : (i : ι) → Monoid (M i)
      inst✝ : (i : ι) → DecidableEq (M i)
      i : ι
      m : M i
      w : Monoid.CoprodI.Word M
      h : Ne w.fstIdx (Option.some i)
      m' : M i
      w' : Monoid.CoprodI.Word M
      h' : Ne w'.fstIdx (Option.some i)
      he : Eq (Monoid.CoprodI.Word.rcons { head := m, tail := w, fstIdx_ne := h }) ( …
      hm : Eq m 1
      hm' : Not (Eq m' 1)
      ⊢ False
    -/
    simp only [rcons, dif_pos hm, dif_neg hm'] at he
    /-
      case neg
      ι : Type u_1
      M : ι → Type u_2
      inst✝¹ : (i : ι) → Monoid (M i)
      inst✝ : (i : ι) → DecidableEq (M i)
      i : ι
      m : M i
      w : Monoid.CoprodI.Word M
      h : Ne w.fstIdx (Option.some i)
      m' : M i
      w' : Monoid.CoprodI.Word M
      h' : Ne w'.fstIdx (Option.some i)
      hm : Eq m 1
      hm' : Not (Eq m' 1)
      he : Eq w (Monoid.CoprodI.Word.cons m' w' ⋯ ⋯)
      ⊢ False
    -/
    rw [he] at h
    /-
      case neg
      ι : Type u_1
      M : ι → Type u_2
      inst✝¹ : (i : ι) → Monoid (M i)
      inst✝ : (i : ι) → DecidableEq (M i)
      i : ι
      m : M i
      w : Monoid.CoprodI.Word M
      m' : M i
      w' : Monoid.CoprodI.Word M
      h' : Ne w'.fstIdx (Option.some i)
      hm : Eq m 1
      hm' : Not (Eq m' 1)
      h : Ne (Monoid.CoprodI.Word.cons m' w' ⋯ ⋯).fstIdx (Option.some i)
      he : Eq w (Monoid.CoprodI.Word.cons m' w' ⋯ ⋯)
      ⊢ False
    -/
    exact h rfl
    /-
      🎉 no goals
    -/
    /-
      case pos
      ι : Type u_1
      M : ι → Type u_2
      inst✝¹ : (i : ι) → Monoid (M i)
      inst✝ : (i : ι) → DecidableEq (M i)
      i : ι
      m : M i
      w : Monoid.CoprodI.Word M
      h : Ne w.fstIdx (Option.some i)
      m' : M i
      w' : Monoid.CoprodI.Word M
      h' : Ne w'.fstIdx (Option.some i)
      he : Eq (Monoid.CoprodI.Word.rcons { head := m, tail := w, fstIdx_ne := h }) ( …
      hm : Not (Eq m 1)
      hm' : Eq m' 1
      ⊢ Eq { head := m, tail := w, fstIdx_ne := h } { head := m', tail := w', fstIdx …
    -/
  · exfalso
    /-
      case pos
      ι : Type u_1
      M : ι → Type u_2
      inst✝¹ : (i : ι) → Monoid (M i)
      inst✝ : (i : ι) → DecidableEq (M i)
      i : ι
      m : M i
      w : Monoid.CoprodI.Word M
      h : Ne w.fstIdx (Option.some i)
      m' : M i
      w' : Monoid.CoprodI.Word M
      h' : Ne w'.fstIdx (Option.some i)
      he : Eq (Monoid.CoprodI.Word.rcons { head := m, tail := w, fstIdx_ne := h }) ( …
      hm : Not (Eq m 1)
      hm' : Eq m' 1
      ⊢ False
    -/
    simp only [rcons, dif_pos hm', dif_neg hm] at he
    /-
      case pos
      ι : Type u_1
      M : ι → Type u_2
      inst✝¹ : (i : ι) → Monoid (M i)
      inst✝ : (i : ι) → DecidableEq (M i)
      i : ι
      m : M i
      w : Monoid.CoprodI.Word M
      h : Ne w.fstIdx (Option.some i)
      m' : M i
      w' : Monoid.CoprodI.Word M
      h' : Ne w'.fstIdx (Option.some i)
      hm : Not (Eq m 1)
      hm' : Eq m' 1
      he : Eq (Monoid.CoprodI.Word.cons m w ⋯ ⋯) w'
      ⊢ False
    -/
    rw [← he] at h'
    /-
      case pos
      ι : Type u_1
      M : ι → Type u_2
      inst✝¹ : (i : ι) → Monoid (M i)
      inst✝ : (i : ι) → DecidableEq (M i)
      i : ι
      m : M i
      w : Monoid.CoprodI.Word M
      h : Ne w.fstIdx (Option.some i)
      m' : M i
      w' : Monoid.CoprodI.Word M
      hm : Not (Eq m 1)
      h' : Ne (Monoid.CoprodI.Word.cons m w ⋯ ⋯).fstIdx (Option.some i)
      hm' : Eq m' 1
      he : Eq (Monoid.CoprodI.Word.cons m w ⋯ ⋯) w'
      ⊢ False
    -/
    exact h' rfl
    /-
      🎉 no goals
    -/
  · have : m = m' ∧ w.toList = w'.toList := by
      simpa [cons, rcons, dif_neg hm, dif_neg hm', eq_self_iff_true, Subtype.mk_eq_mk,
        heq_iff_eq, ← Subtype.ext_iff_val] using he
    /-
      case neg
      ι : Type u_1
      M : ι → Type u_2
      inst✝¹ : (i : ι) → Monoid (M i)
      inst✝ : (i : ι) → DecidableEq (M i)
      i : ι
      m : M i
      w : Monoid.CoprodI.Word M
      h : Ne w.fstIdx (Option.some i)
      m' : M i
      w' : Monoid.CoprodI.Word M
      h' : Ne w'.fstIdx (Option.some i)
      he : Eq (Monoid.CoprodI.Word.rcons { head := m, tail := w, fstIdx_ne := h }) ( …
      hm : Not (Eq m 1)
      hm' : Not (Eq m' 1)
      this : And (Eq m m') (Eq w.toList w'.toList)
      ⊢ Eq { head := m, tail := w, fstIdx_ne := h } { head := m', tail := w', fstIdx …
    -/
    rcases this with ⟨rfl, h⟩
    /-
      case neg.intro
      ι : Type u_1
      M : ι → Type u_2
      inst✝¹ : (i : ι) → Monoid (M i)
      inst✝ : (i : ι) → DecidableEq (M i)
      i : ι
      m : M i
      w : Monoid.CoprodI.Word M
      h✝ : Ne w.fstIdx (Option.some i)
      w' : Monoid.CoprodI.Word M
      h' : Ne w'.fstIdx (Option.some i)
      hm : Not (Eq m 1)
      h : Eq w.toList w'.toList
      he : Eq (Monoid.CoprodI.Word.rcons { head := m, tail := w, fstIdx_ne := h✝ })  …
      hm' : Not (Eq m 1)
      ⊢ Eq { head := m, tail := w, fstIdx_ne := h✝ } { head := m, tail := w', fstIdx …
    -/
    congr
    /-
      case neg.intro.e_tail
      ι : Type u_1
      M : ι → Type u_2
      inst✝¹ : (i : ι) → Monoid (M i)
      inst✝ : (i : ι) → DecidableEq (M i)
      i : ι
      m : M i
      w : Monoid.CoprodI.Word M
      h✝ : Ne w.fstIdx (Option.some i)
      w' : Monoid.CoprodI.Word M
      h' : Ne w'.fstIdx (Option.some i)
      hm : Not (Eq m 1)
      h : Eq w.toList w'.toList
      he : Eq (Monoid.CoprodI.Word.rcons { head := m, tail := w, fstIdx_ne := h✝ })  …
      hm' : Not (Eq m 1)
      ⊢ Eq w w'
    -/
    exact Word.ext h
    /-
      🎉 no goals
    -/


theorem mem_rcons_iff {i j : ι} (p : Pair M i) (m : M j) :
    ⟨_, m⟩ ∈ (rcons p).toList ↔ ⟨_, m⟩ ∈ p.tail.toList ∨
      m ≠ 1 ∧ (∃ h : i = j, m = h ▸ p.head) := by
  /-
    ι : Type u_1
    M : ι → Type u_2
    inst✝¹ : (i : ι) → Monoid (M i)
    inst✝ : (i : ι) → DecidableEq (M i)
    i j : ι
    p : Monoid.CoprodI.Word.Pair M i
    m : M j
    ⊢ Iff (Membership.mem (Monoid.CoprodI.Word.rcons p).toList ⟨j, m⟩) (Or (Member …
  -/
  simp only [rcons, cons, ne_eq]
  /-
    ι : Type u_1
    M : ι → Type u_2
    inst✝¹ : (i : ι) → Monoid (M i)
    inst✝ : (i : ι) → DecidableEq (M i)
    i j : ι
    p : Monoid.CoprodI.Word.Pair M i
    m : M j
    ⊢ Iff (Membership.mem (dite (Eq p.head 1) (fun h => p.tail) fun h => { toList  …
  -/
  by_cases hij : i = j
    /-
      case pos
      ι : Type u_1
      M : ι → Type u_2
      inst✝¹ : (i : ι) → Monoid (M i)
      inst✝ : (i : ι) → DecidableEq (M i)
      i j : ι
      p : Monoid.CoprodI.Word.Pair M i
      m : M j
      hij : Eq i j
      ⊢ Iff (Membership.mem (dite (Eq p.head 1) (fun h => p.tail) fun h => { toList  …
    -/
  · subst i
    /-
      case pos
      ι : Type u_1
      M : ι → Type u_2
      inst✝¹ : (i : ι) → Monoid (M i)
      inst✝ : (i : ι) → DecidableEq (M i)
      j : ι
      m : M j
      p : Monoid.CoprodI.Word.Pair M j
      ⊢ Iff (Membership.mem (dite (Eq p.head 1) (fun h => p.tail) fun h => { toList  …
    -/
    by_cases hm : m = p.head
      /-
        case pos
        ι : Type u_1
        M : ι → Type u_2
        inst✝¹ : (i : ι) → Monoid (M i)
        inst✝ : (i : ι) → DecidableEq (M i)
        j : ι
        m : M j
        p : Monoid.CoprodI.Word.Pair M j
        hm : Eq m p.head
        ⊢ Iff (Membership.mem (dite (Eq p.head 1) (fun h => p.tail) fun h => { toList  …
      -/
    · subst m
      /-
        case pos
        ι : Type u_1
        M : ι → Type u_2
        inst✝¹ : (i : ι) → Monoid (M i)
        inst✝ : (i : ι) → DecidableEq (M i)
        j : ι
        p : Monoid.CoprodI.Word.Pair M j
        ⊢ Iff (Membership.mem (dite (Eq p.head 1) (fun h => p.tail) fun h => { toList  …
      -/
                    /-
                      🎉 no goals
                    -/
      split_ifs <;> simp_all
                    /-
                      🎉 no goals
                    -/
      /-
        case neg
        ι : Type u_1
        M : ι → Type u_2
        inst✝¹ : (i : ι) → Monoid (M i)
        inst✝ : (i : ι) → DecidableEq (M i)
        j : ι
        m : M j
        p : Monoid.CoprodI.Word.Pair M j
        hm : Not (Eq m p.head)
        ⊢ Iff (Membership.mem (dite (Eq p.head 1) (fun h => p.tail) fun h => { toList  …
      -/
                    /-
                      🎉 no goals
                    -/
    · split_ifs <;> simp_all
                    /-
                      🎉 no goals
                    -/
    /-
      case neg
      ι : Type u_1
      M : ι → Type u_2
      inst✝¹ : (i : ι) → Monoid (M i)
      inst✝ : (i : ι) → DecidableEq (M i)
      i j : ι
      p : Monoid.CoprodI.Word.Pair M i
      m : M j
      hij : Not (Eq i j)
      ⊢ Iff (Membership.mem (dite (Eq p.head 1) (fun h => p.tail) fun h => { toList  …
    -/
                  /-
                    🎉 no goals
                  -/
  · split_ifs <;> simp_all [Ne.symm hij]
                  /-
                    🎉 no goals
                  -/


/-- Induct on a word by adding letters one at a time without reduction,
effectively inducting on the underlying `List`. -/
@[elab_as_elim]
def consRecOn {motive : Word M → Sort*} (w : Word M) (h_empty : motive empty)
    (h_cons : ∀ (i) (m : M i) (w) h1 h2, motive w → motive (cons m w h1 h2)) :
    motive w := by
  /-
    ι : Type u_1
    M : ι → Type u_2
    inst✝¹ : (i : ι) → Monoid (M i)
    N : Type u_3
    inst✝ : Monoid N
    motive : Monoid.CoprodI.Word M → Sort u_4
    w : Monoid.CoprodI.Word M
    h_empty : motive Monoid.CoprodI.Word.empty
    h_cons : (i : ι) → (m : M i) → (w : Monoid.CoprodI.Word M) → (h1 : Ne w.fstIdx …
    ⊢ motive w
  -/
  rcases w with ⟨w, h1, h2⟩
  induction w with
  | nil => exact h_empty
  | cons m w ih =>
    refine h_cons m.1 m.2 ⟨w, fun _ hl => h1 _ (List.mem_cons_of_mem _ hl), h2.tail⟩ ?_ ?_ (ih _ _)
    · rw [List.chain'_cons'] at h2
      simp only [fstIdx, ne_eq, Option.map_eq_some',
        Sigma.exists, exists_and_right, exists_eq_right, not_exists]
      intro m' hm'
      exact h2.1 _ hm' rfl
    · exact h1 _ (List.mem_cons_self _ _)


@[simp]
theorem consRecOn_empty {motive : Word M → Sort*} (h_empty : motive empty)
    (h_cons : ∀ (i) (m : M i) (w) h1 h2, motive w → motive (cons m w h1 h2)) :
    consRecOn empty h_empty h_cons = h_empty := rfl


@[simp]
theorem consRecOn_cons {motive : Word M → Sort*} (i) (m : M i) (w : Word M) h1 h2
    (h_empty : motive empty)
    (h_cons : ∀ (i) (m : M i) (w) h1 h2, motive w → motive (cons m w h1 h2)) :
    consRecOn (cons m w h1 h2) h_empty h_cons = h_cons i m w h1 h2
      (consRecOn w h_empty h_cons) := rfl


/-- Given `i : ι`, any reduced word can be decomposed into a pair `p` such that `w = rcons p`. -/
private def equivPairAux (i) (w : Word M) : { p : Pair M i // rcons p = w } :=
                              /-
                                ι : Type u_1
                                M : ι → Type u_2
                                inst✝³ : (i : ι) → Monoid (M i)
                                N : Type u_3
                                inst✝² : Monoid N
                                inst✝¹ : DecidableEq ι
                                inst✝ : (i : ι) → DecidableEq (M i)
                                i : ι
                                w : Monoid.CoprodI.Word M
                                ⊢ Ne Monoid.CoprodI.Word.empty.fstIdx (Option.some i)
                              -/
                              /-
                                🎉 no goals
                              -/
  consRecOn w ⟨⟨1, .empty, by simp [fstIdx, empty]⟩, by simp [rcons]⟩ <|
                                                        /-
                                                          🎉 no goals
                                                        -/
    fun j m w h1 h2 _ =>
      if ij : i = j then
        { val :=
          { head := ij ▸ m
            tail := w
            fstIdx_ne := ij ▸ h1 }
                         /-
                           ι : Type u_1
                           M : ι → Type u_2
                           inst✝³ : (i : ι) → Monoid (M i)
                           N : Type u_3
                           inst✝² : Monoid N
                           inst✝¹ : DecidableEq ι
                           inst✝ : (i : ι) → DecidableEq (M i)
                           i : ι
                           w✝ : Monoid.CoprodI.Word M
                           j : ι
                           m : M j
                           w : Monoid.CoprodI.Word M
                           h1 : Ne w.fstIdx (Option.some j)
                           h2 : Ne m 1
                           x✝ : Subtype fun p => Eq (Monoid.CoprodI.Word.rcons p) w
                           ij : Eq i j
                           ⊢ Eq (Monoid.CoprodI.Word.rcons { head := Eq.rec m ⋯, tail := w, fstIdx_ne :=  …
                         -/
          property := by subst ij; simp [rcons, h2] }
                                   /-
                                     🎉 no goals
                                   -/
                                   /-
                                     ι : Type u_1
                                     M : ι → Type u_2
                                     inst✝³ : (i : ι) → Monoid (M i)
                                     N : Type u_3
                                     inst✝² : Monoid N
                                     inst✝¹ : DecidableEq ι
                                     inst✝ : (i : ι) → DecidableEq (M i)
                                     i : ι
                                     w✝ : Monoid.CoprodI.Word M
                                     j : ι
                                     m : M j
                                     w : Monoid.CoprodI.Word M
                                     h1 : Ne w.fstIdx (Option.some j)
                                     h2 : Ne m 1
                                     x✝ : Subtype fun p => Eq (Monoid.CoprodI.Word.rcons p) w
                                     ij : Not (Eq i j)
                                     ⊢ Ne (Monoid.CoprodI.Word.cons m w h1 h2).fstIdx (Option.some i)
                                   -/
                                   /-
                                     🎉 no goals
                                   -/
      else ⟨⟨1, cons m w h1 h2, by simp [cons, fstIdx, Ne.symm ij]⟩,  by simp [rcons]⟩
                                                                         /-
                                                                           🎉 no goals
                                                                         -/


/-- The equivalence between words and pairs. Given a word, it decomposes it as a pair by removing
the first letter if it comes from `M i`. Given a pair, it prepends the head to the tail. -/
def equivPair (i) : Word M ≃ Pair M i where
  toFun w := (equivPairAux i w).val
  invFun := rcons
  left_inv w := (equivPairAux i w).property
  right_inv _ := rcons_inj (equivPairAux i _).property


theorem equivPair_symm (i) (p : Pair M i) : (equivPair i).symm p = rcons p :=
  rfl


theorem equivPair_eq_of_fstIdx_ne {i} {w : Word M} (h : fstIdx w ≠ some i) :
    equivPair i w = ⟨1, w, h⟩ :=
  (equivPair i).apply_eq_iff_eq_symm_apply.mpr <| Eq.symm (dif_pos rfl)


theorem mem_equivPair_tail_iff {i j : ι} {w : Word M} (m : M i) :
    (⟨i, m⟩ ∈ (equivPair j w).tail.toList) ↔ ⟨i, m⟩ ∈ w.toList.tail
      ∨ i ≠ j ∧ ∃ h : w.toList ≠ [], w.toList.head h = ⟨i, m⟩ := by
  /-
    ι : Type u_1
    M : ι → Type u_2
    inst✝² : (i : ι) → Monoid (M i)
    inst✝¹ : DecidableEq ι
    inst✝ : (i : ι) → DecidableEq (M i)
    i j : ι
    w : Monoid.CoprodI.Word M
    m : M i
    ⊢ Iff (Membership.mem ((Monoid.CoprodI.Word.equivPair j) w).tail.toList ⟨i, m⟩ …
  -/
  simp only [equivPair, equivPairAux, ne_eq, Equiv.coe_fn_mk]
  induction w using consRecOn with
  | h_empty => simp
  | h_cons k g tail h1 h2 ih =>
    simp only [consRecOn_cons]
    split_ifs with h
    · subst k
      by_cases hij : j = i <;> simp_all
    · by_cases hik : i = k
      · subst i; simp_all [@eq_comm _ m g, @eq_comm _ k j, or_comm]
      · simp [hik, Ne.symm hik]


theorem mem_of_mem_equivPair_tail {i j : ι} {w : Word M} (m : M i) :
    (⟨i, m⟩ ∈ (equivPair j w).tail.toList) → ⟨i, m⟩ ∈ w.toList := by
  /-
    ι : Type u_1
    M : ι → Type u_2
    inst✝² : (i : ι) → Monoid (M i)
    inst✝¹ : DecidableEq ι
    inst✝ : (i : ι) → DecidableEq (M i)
    i j : ι
    w : Monoid.CoprodI.Word M
    m : M i
    ⊢ Membership.mem ((Monoid.CoprodI.Word.equivPair j) w).tail.toList ⟨i, m⟩ → Me …
  -/
  rw [mem_equivPair_tail_iff]
  /-
    ι : Type u_1
    M : ι → Type u_2
    inst✝² : (i : ι) → Monoid (M i)
    inst✝¹ : DecidableEq ι
    inst✝ : (i : ι) → DecidableEq (M i)
    i j : ι
    w : Monoid.CoprodI.Word M
    m : M i
    ⊢ Or (Membership.mem w.toList.tail ⟨i, m⟩) (And (Ne i j) (Exists fun h => Eq ( …
  -/
  rintro (h | h)
    /-
      case inl
      ι : Type u_1
      M : ι → Type u_2
      inst✝² : (i : ι) → Monoid (M i)
      inst✝¹ : DecidableEq ι
      inst✝ : (i : ι) → DecidableEq (M i)
      i j : ι
      w : Monoid.CoprodI.Word M
      m : M i
      h : Membership.mem w.toList.tail ⟨i, m⟩
      ⊢ Membership.mem w.toList ⟨i, m⟩
    -/
  · exact List.mem_of_mem_tail h
    /-
      🎉 no goals
    -/
    /-
      case inr
      ι : Type u_1
      M : ι → Type u_2
      inst✝² : (i : ι) → Monoid (M i)
      inst✝¹ : DecidableEq ι
      inst✝ : (i : ι) → DecidableEq (M i)
      i j : ι
      w : Monoid.CoprodI.Word M
      m : M i
      h : And (Ne i j) (Exists fun h => Eq (w.toList.head h) ⟨i, m⟩)
      ⊢ Membership.mem w.toList ⟨i, m⟩
    -/
                                 /-
                                   🎉 no goals
                                 -/
  · revert h; cases w.toList <;> simp (config := {contextual := true})
                                 /-
                                   🎉 no goals
                                 -/


theorem equivPair_head {i : ι} {w : Word M} :
    (equivPair i w).head =
      if h : ∃ (h : w.toList ≠ []), (w.toList.head h).1 = i
      then h.snd ▸ (w.toList.head h.1).2
      else 1 := by
  /-
    ι : Type u_1
    M : ι → Type u_2
    inst✝² : (i : ι) → Monoid (M i)
    inst✝¹ : DecidableEq ι
    inst✝ : (i : ι) → DecidableEq (M i)
    i : ι
    w : Monoid.CoprodI.Word M
    ⊢ Eq ((Monoid.CoprodI.Word.equivPair i) w).head (dite (Exists fun h => Eq (w.t …
  -/
  simp only [equivPair, equivPairAux]
  induction w using consRecOn with
  | h_empty => simp
  | h_cons head =>
    by_cases hi : i = head
    · subst hi; simp
    · simp [hi, Ne.symm hi]


instance summandAction (i) : MulAction (M i) (Word M) where
  smul m w := rcons { equivPair i w with head := m * (equivPair i w).head }
  one_smul w := by
    /-
      ι : Type u_1
      M : ι → Type u_2
      inst✝³ : (i : ι) → Monoid (M i)
      N : Type u_3
      inst✝² : Monoid N
      inst✝¹ : DecidableEq ι
      inst✝ : (i : ι) → DecidableEq (M i)
      i : ι
      w : Monoid.CoprodI.Word M
      ⊢ Eq (HSMul.hSMul 1 w) w
    -/
    apply (equivPair i).symm_apply_eq.mpr
    /-
      ι : Type u_1
      M : ι → Type u_2
      inst✝³ : (i : ι) → Monoid (M i)
      N : Type u_3
      inst✝² : Monoid N
      inst✝¹ : DecidableEq ι
      inst✝ : (i : ι) → DecidableEq (M i)
      i : ι
      w : Monoid.CoprodI.Word M
      ⊢ Eq
          (let __src := (Monoid.CoprodI.Word.equivPair i) w;
          { head := HMul.hMul 1 ((Monoid.CoprodI.Word.equivPair i) w).head, tail :=  …
          ((Monoid.CoprodI.Word.equivPair i) w)
    -/
    simp [equivPair]
    /-
      🎉 no goals
    -/
  mul_smul m m' w := by
    /-
      ι : Type u_1
      M : ι → Type u_2
      inst✝³ : (i : ι) → Monoid (M i)
      N : Type u_3
      inst✝² : Monoid N
      inst✝¹ : DecidableEq ι
      inst✝ : (i : ι) → DecidableEq (M i)
      i : ι
      m m' : M i
      w : Monoid.CoprodI.Word M
      ⊢ Eq (HSMul.hSMul (HMul.hMul m m') w) (HSMul.hSMul m (HSMul.hSMul m' w))
    -/
    dsimp [instHSMul]
    /-
      ι : Type u_1
      M : ι → Type u_2
      inst✝³ : (i : ι) → Monoid (M i)
      N : Type u_3
      inst✝² : Monoid N
      inst✝¹ : DecidableEq ι
      inst✝ : (i : ι) → DecidableEq (M i)
      i : ι
      m m' : M i
      w : Monoid.CoprodI.Word M
      ⊢ Eq (Monoid.CoprodI.Word.rcons { head := HMul.hMul (HMul.hMul m m') ((Monoid. …
    -/
    simp [mul_assoc, ← equivPair_symm, Equiv.apply_symm_apply]
    /-
      🎉 no goals
    -/


instance : MulAction (CoprodI M) (Word M) :=
  MulAction.ofEndHom (lift fun _ => MulAction.toEndHom)


theorem smul_def {i} (m : M i) (w : Word M) :
    m • w = rcons { equivPair i w with head := m * (equivPair i w).head } :=
  rfl


theorem of_smul_def (i) (w : Word M) (m : M i) :
    of m • w = rcons { equivPair i w with head := m * (equivPair i w).head } :=
  rfl


theorem equivPair_smul_same {i} (m : M i) (w : Word M) :
    equivPair i (of m • w) = ⟨m * (equivPair i w).head, (equivPair i w).tail,
      (equivPair i w).fstIdx_ne⟩ := by
  /-
    ι : Type u_1
    M : ι → Type u_2
    inst✝² : (i : ι) → Monoid (M i)
    inst✝¹ : DecidableEq ι
    inst✝ : (i : ι) → DecidableEq (M i)
    i : ι
    m : M i
    w : Monoid.CoprodI.Word M
    ⊢ Eq ((Monoid.CoprodI.Word.equivPair i) (HSMul.hSMul (Monoid.CoprodI.of m) w)) …
  -/
  rw [of_smul_def, ← equivPair_symm]
  /-
    ι : Type u_1
    M : ι → Type u_2
    inst✝² : (i : ι) → Monoid (M i)
    inst✝¹ : DecidableEq ι
    inst✝ : (i : ι) → DecidableEq (M i)
    i : ι
    m : M i
    w : Monoid.CoprodI.Word M
    ⊢ Eq
        ((Monoid.CoprodI.Word.equivPair i)
          ((Monoid.CoprodI.Word.equivPair i).symm
            (let __src := (Monoid.CoprodI.Word.equivPair i) w;
            { head := HMul.hMul m ((Monoid.CoprodI.Word.equivPair i) w).head, tail …
        { head := HMul.hMul m ((Monoid.CoprodI.Word.equivPair i) w).head, tail :=  …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem equivPair_tail {i} (p : Pair M i) :
    equivPair i p.tail = ⟨1, p.tail, p.fstIdx_ne⟩ :=
  equivPair_eq_of_fstIdx_ne _


theorem smul_eq_of_smul {i} (m : M i) (w : Word M) :
    m • w = of m • w := rfl


theorem mem_smul_iff {i j : ι} {m₁ : M i} {m₂ : M j} {w : Word M} :
    ⟨_, m₁⟩ ∈ (of m₂ • w).toList ↔
      (¬i = j ∧ ⟨i, m₁⟩ ∈ w.toList)
      ∨ (m₁ ≠ 1 ∧ ∃ (hij : i = j),(⟨i, m₁⟩ ∈ w.toList.tail) ∨
        (∃ m', ⟨j, m'⟩ ∈ w.toList.head? ∧ m₁ = hij ▸ (m₂ * m')) ∨
        (w.fstIdx ≠ some j ∧ m₁ = hij ▸ m₂)) := by
  /-
    ι : Type u_1
    M : ι → Type u_2
    inst✝² : (i : ι) → Monoid (M i)
    inst✝¹ : DecidableEq ι
    inst✝ : (i : ι) → DecidableEq (M i)
    i j : ι
    m₁ : M i
    m₂ : M j
    w : Monoid.CoprodI.Word M
    ⊢ Iff (Membership.mem (HSMul.hSMul (Monoid.CoprodI.of m₂) w).toList ⟨i, m₁⟩) ( …
  -/
  rw [of_smul_def, mem_rcons_iff, mem_equivPair_tail_iff, equivPair_head, or_assoc]
  /-
    ι : Type u_1
    M : ι → Type u_2
    inst✝² : (i : ι) → Monoid (M i)
    inst✝¹ : DecidableEq ι
    inst✝ : (i : ι) → DecidableEq (M i)
    i j : ι
    m₁ : M i
    m₂ : M j
    w : Monoid.CoprodI.Word M
    ⊢ Iff
        (Or (Membership.mem w.toList.tail ⟨i, m₁⟩)
          (Or (And (Ne i j) (Exists fun h => Eq (w.toList.head h) ⟨i, m₁⟩))
            (And (Ne m₁ 1)
              (Exists fun h =>
                Eq m₁
                  (Eq.rec
                    (let __src := (Monoid.CoprodI.Word.equivPair j) w;
                      { head := HMul.hMul m₂ (dite (Exists fun h => Eq (w.toList.h …
                    h)))))
        (Or (And (Not (Eq i j)) (Membership.mem w.toList ⟨i, m₁⟩)) (And (Ne m₁ 1)  …
  -/
  by_cases hij : i = j
    /-
      case pos
      ι : Type u_1
      M : ι → Type u_2
      inst✝² : (i : ι) → Monoid (M i)
      inst✝¹ : DecidableEq ι
      inst✝ : (i : ι) → DecidableEq (M i)
      i j : ι
      m₁ : M i
      m₂ : M j
      w : Monoid.CoprodI.Word M
      hij : Eq i j
      ⊢ Iff
          (Or (Membership.mem w.toList.tail ⟨i, m₁⟩)
            (Or (And (Ne i j) (Exists fun h => Eq (w.toList.head h) ⟨i, m₁⟩))
              (And (Ne m₁ 1)
                (Exists fun h =>
                  Eq m₁
                    (Eq.rec
                      (let __src := (Monoid.CoprodI.Word.equivPair j) w;
                        { head := HMul.hMul m₂ (dite (Exists fun h => Eq (w.toList.h …
                      h)))))
          (Or (And (Not (Eq i j)) (Membership.mem w.toList ⟨i, m₁⟩)) (And (Ne m₁ 1)  …
    -/
  · subst i
    /-
      case pos
      ι : Type u_1
      M : ι → Type u_2
      inst✝² : (i : ι) → Monoid (M i)
      inst✝¹ : DecidableEq ι
      inst✝ : (i : ι) → DecidableEq (M i)
      j : ι
      m₂ : M j
      w : Monoid.CoprodI.Word M
      m₁ : M j
      ⊢ Iff
          (Or (Membership.mem w.toList.tail ⟨j, m₁⟩)
            (Or (And (Ne j j) (Exists fun h => Eq (w.toList.head h) ⟨j, m₁⟩))
              (And (Ne m₁ 1)
                (Exists fun h =>
                  Eq m₁
                    (Eq.rec
                      (let __src := (Monoid.CoprodI.Word.equivPair j) w;
                        { head := HMul.hMul m₂ (dite (Exists fun h => Eq (w.toList.h …
                      h)))))
          (Or (And (Not (Eq j j)) (Membership.mem w.toList ⟨j, m₁⟩)) (And (Ne m₁ 1)  …
    -/
    simp only [not_true, ne_eq, false_and, exists_prop, true_and, false_or]
    /-
      case pos
      ι : Type u_1
      M : ι → Type u_2
      inst✝² : (i : ι) → Monoid (M i)
      inst✝¹ : DecidableEq ι
      inst✝ : (i : ι) → DecidableEq (M i)
      j : ι
      m₂ : M j
      w : Monoid.CoprodI.Word M
      m₁ : M j
      ⊢ Iff (Or (Membership.mem w.toList.tail ⟨j, m₁⟩) (And (Not (Eq m₁ 1)) (Eq m₁ ( …
    -/
    by_cases hw : ⟨j, m₁⟩ ∈ w.toList.tail
      /-
        case pos
        ι : Type u_1
        M : ι → Type u_2
        inst✝² : (i : ι) → Monoid (M i)
        inst✝¹ : DecidableEq ι
        inst✝ : (i : ι) → DecidableEq (M i)
        j : ι
        m₂ : M j
        w : Monoid.CoprodI.Word M
        m₁ : M j
        hw : Membership.mem w.toList.tail ⟨j, m₁⟩
        ⊢ Iff (Or (Membership.mem w.toList.tail ⟨j, m₁⟩) (And (Not (Eq m₁ 1)) (Eq m₁ ( …
      -/
    · simp [hw, show m₁ ≠ 1 from w.ne_one _ (List.mem_of_mem_tail hw)]
      /-
        🎉 no goals
      -/
      /-
        case neg
        ι : Type u_1
        M : ι → Type u_2
        inst✝² : (i : ι) → Monoid (M i)
        inst✝¹ : DecidableEq ι
        inst✝ : (i : ι) → DecidableEq (M i)
        j : ι
        m₂ : M j
        w : Monoid.CoprodI.Word M
        m₁ : M j
        hw : Not (Membership.mem w.toList.tail ⟨j, m₁⟩)
        ⊢ Iff (Or (Membership.mem w.toList.tail ⟨j, m₁⟩) (And (Not (Eq m₁ 1)) (Eq m₁ ( …
      -/
    · simp only [hw, false_or, Option.mem_def, ne_eq, and_congr_right_iff]
      /-
        case neg
        ι : Type u_1
        M : ι → Type u_2
        inst✝² : (i : ι) → Monoid (M i)
        inst✝¹ : DecidableEq ι
        inst✝ : (i : ι) → DecidableEq (M i)
        j : ι
        m₂ : M j
        w : Monoid.CoprodI.Word M
        m₁ : M j
        hw : Not (Membership.mem w.toList.tail ⟨j, m₁⟩)
        ⊢ Not (Eq m₁ 1) → Iff (Eq m₁ (HMul.hMul m₂ (dite (Exists fun h => Eq (w.toList …
      -/
      intro hm1
      /-
        case neg
        ι : Type u_1
        M : ι → Type u_2
        inst✝² : (i : ι) → Monoid (M i)
        inst✝¹ : DecidableEq ι
        inst✝ : (i : ι) → DecidableEq (M i)
        j : ι
        m₂ : M j
        w : Monoid.CoprodI.Word M
        m₁ : M j
        hw : Not (Membership.mem w.toList.tail ⟨j, m₁⟩)
        hm1 : Not (Eq m₁ 1)
        ⊢ Iff (Eq m₁ (HMul.hMul m₂ (dite (Exists fun h => Eq (w.toList.head h).fst j)  …
      -/
      split_ifs with h
        /-
          case pos
          ι : Type u_1
          M : ι → Type u_2
          inst✝² : (i : ι) → Monoid (M i)
          inst✝¹ : DecidableEq ι
          inst✝ : (i : ι) → DecidableEq (M i)
          j : ι
          m₂ : M j
          w : Monoid.CoprodI.Word M
          m₁ : M j
          hw : Not (Membership.mem w.toList.tail ⟨j, m₁⟩)
          hm1 : Not (Eq m₁ 1)
          h : Exists fun h => Eq (w.toList.head h).fst j
          ⊢ Iff (Eq m₁ (HMul.hMul m₂ (Eq.rec (w.toList.head ⋯).snd ⋯))) (Or (Exists fun  …
        -/
      · rcases h with ⟨hnil, rfl⟩
        /-
          case pos.intro
          ι : Type u_1
          M : ι → Type u_2
          inst✝² : (i : ι) → Monoid (M i)
          inst✝¹ : DecidableEq ι
          inst✝ : (i : ι) → DecidableEq (M i)
          w : Monoid.CoprodI.Word M
          hnil : Not (Eq w.toList List.nil)
          m₂ m₁ : M (w.toList.head hnil).fst
          hw : Not (Membership.mem w.toList.tail ⟨(w.toList.head hnil).fst, m₁⟩)
          hm1 : Not (Eq m₁ 1)
          ⊢ Iff (Eq m₁ (HMul.hMul m₂ (Eq.rec (w.toList.head ⋯).snd ⋯))) (Or (Exists fun  …
        -/
        simp only [List.head?_eq_head hnil, Option.some.injEq, ne_eq]
        /-
          case pos.intro
          ι : Type u_1
          M : ι → Type u_2
          inst✝² : (i : ι) → Monoid (M i)
          inst✝¹ : DecidableEq ι
          inst✝ : (i : ι) → DecidableEq (M i)
          w : Monoid.CoprodI.Word M
          hnil : Not (Eq w.toList List.nil)
          m₂ m₁ : M (w.toList.head hnil).fst
          hw : Not (Membership.mem w.toList.tail ⟨(w.toList.head hnil).fst, m₁⟩)
          hm1 : Not (Eq m₁ 1)
          ⊢ Iff (Eq m₁ (HMul.hMul m₂ (w.toList.head ⋯).snd)) (Or (Exists fun m' => And ( …
        -/
        constructor
          /-
            case pos.intro.mp
            ι : Type u_1
            M : ι → Type u_2
            inst✝² : (i : ι) → Monoid (M i)
            inst✝¹ : DecidableEq ι
            inst✝ : (i : ι) → DecidableEq (M i)
            w : Monoid.CoprodI.Word M
            hnil : Not (Eq w.toList List.nil)
            m₂ m₁ : M (w.toList.head hnil).fst
            hw : Not (Membership.mem w.toList.tail ⟨(w.toList.head hnil).fst, m₁⟩)
            hm1 : Not (Eq m₁ 1)
            ⊢ Eq m₁ (HMul.hMul m₂ (w.toList.head ⋯).snd) → Or (Exists fun m' => And (Eq (w …
          -/
        · rintro rfl
          /-
            case pos.intro.mp
            ι : Type u_1
            M : ι → Type u_2
            inst✝² : (i : ι) → Monoid (M i)
            inst✝¹ : DecidableEq ι
            inst✝ : (i : ι) → DecidableEq (M i)
            w : Monoid.CoprodI.Word M
            hnil : Not (Eq w.toList List.nil)
            m₂ : M (w.toList.head hnil).fst
            hw : Not (Membership.mem w.toList.tail ⟨(w.toList.head hnil).fst, HMul.hMul m₂ …
            hm1 : Not (Eq (HMul.hMul m₂ (w.toList.head ⋯).snd) 1)
            ⊢ Or (Exists fun m' => And (Eq (w.toList.head hnil) ⟨(w.toList.head hnil).fst, …
          -/
          exact Or.inl ⟨_, rfl, rfl⟩
          /-
            🎉 no goals
          -/
          /-
            case pos.intro.mpr
            ι : Type u_1
            M : ι → Type u_2
            inst✝² : (i : ι) → Monoid (M i)
            inst✝¹ : DecidableEq ι
            inst✝ : (i : ι) → DecidableEq (M i)
            w : Monoid.CoprodI.Word M
            hnil : Not (Eq w.toList List.nil)
            m₂ m₁ : M (w.toList.head hnil).fst
            hw : Not (Membership.mem w.toList.tail ⟨(w.toList.head hnil).fst, m₁⟩)
            hm1 : Not (Eq m₁ 1)
            ⊢ Or (Exists fun m' => And (Eq (w.toList.head hnil) ⟨(w.toList.head hnil).fst, …
          -/
        · rintro (⟨_, h, rfl⟩ | hm')
            /-
              case pos.intro.mpr.inl.intro.intro
              ι : Type u_1
              M : ι → Type u_2
              inst✝² : (i : ι) → Monoid (M i)
              inst✝¹ : DecidableEq ι
              inst✝ : (i : ι) → DecidableEq (M i)
              w : Monoid.CoprodI.Word M
              hnil : Not (Eq w.toList List.nil)
              m₂ w✝ : M (w.toList.head hnil).fst
              h : Eq (w.toList.head hnil) ⟨(w.toList.head hnil).fst, w✝⟩
              hw : Not (Membership.mem w.toList.tail ⟨(w.toList.head hnil).fst, HMul.hMul m₂ …
              hm1 : Not (Eq (HMul.hMul m₂ w✝) 1)
              ⊢ Eq (HMul.hMul m₂ w✝) (HMul.hMul m₂ (w.toList.head ⋯).snd)
            -/
          · simp only [Sigma.ext_iff, heq_eq_eq, true_and] at h
            /-
              case pos.intro.mpr.inl.intro.intro
              ι : Type u_1
              M : ι → Type u_2
              inst✝² : (i : ι) → Monoid (M i)
              inst✝¹ : DecidableEq ι
              inst✝ : (i : ι) → DecidableEq (M i)
              w : Monoid.CoprodI.Word M
              hnil : Not (Eq w.toList List.nil)
              m₂ w✝ : M (w.toList.head hnil).fst
              hw : Not (Membership.mem w.toList.tail ⟨(w.toList.head hnil).fst, HMul.hMul m₂ …
              hm1 : Not (Eq (HMul.hMul m₂ w✝) 1)
              h : Eq (w.toList.head hnil).snd w✝
              ⊢ Eq (HMul.hMul m₂ w✝) (HMul.hMul m₂ (w.toList.head ⋯).snd)
            -/
            subst h
            /-
              case pos.intro.mpr.inl.intro.intro
              ι : Type u_1
              M : ι → Type u_2
              inst✝² : (i : ι) → Monoid (M i)
              inst✝¹ : DecidableEq ι
              inst✝ : (i : ι) → DecidableEq (M i)
              w : Monoid.CoprodI.Word M
              hnil : Not (Eq w.toList List.nil)
              m₂ : M (w.toList.head hnil).fst
              hw : Not (Membership.mem w.toList.tail ⟨(w.toList.head hnil).fst, HMul.hMul m₂ …
              hm1 : Not (Eq (HMul.hMul m₂ (w.toList.head hnil).snd) 1)
              ⊢ Eq (HMul.hMul m₂ (w.toList.head hnil).snd) (HMul.hMul m₂ (w.toList.head ⋯).s …
            -/
            rfl
            /-
              🎉 no goals
            -/
          · simp only [fstIdx, Option.map_eq_some', Sigma.exists,
              exists_and_right, exists_eq_right, not_exists, ne_eq] at hm'
            /-
              case pos.intro.mpr.inr
              ι : Type u_1
              M : ι → Type u_2
              inst✝² : (i : ι) → Monoid (M i)
              inst✝¹ : DecidableEq ι
              inst✝ : (i : ι) → DecidableEq (M i)
              w : Monoid.CoprodI.Word M
              hnil : Not (Eq w.toList List.nil)
              m₂ m₁ : M (w.toList.head hnil).fst
              hw : Not (Membership.mem w.toList.tail ⟨(w.toList.head hnil).fst, m₁⟩)
              hm1 : Not (Eq m₁ 1)
              hm' : And (∀ (x : M (w.toList.head hnil).fst), Not (Eq w.toList.head? (Option. …
              ⊢ Eq m₁ (HMul.hMul m₂ (w.toList.head ⋯).snd)
            -/
            exact (hm'.1 (w.toList.head hnil).2 (by rw [List.head?_eq_head])).elim
            /-
              🎉 no goals
            -/
        /-
          case neg
          ι : Type u_1
          M : ι → Type u_2
          inst✝² : (i : ι) → Monoid (M i)
          inst✝¹ : DecidableEq ι
          inst✝ : (i : ι) → DecidableEq (M i)
          j : ι
          m₂ : M j
          w : Monoid.CoprodI.Word M
          m₁ : M j
          hw : Not (Membership.mem w.toList.tail ⟨j, m₁⟩)
          hm1 : Not (Eq m₁ 1)
          h : Not (Exists fun h => Eq (w.toList.head h).fst j)
          ⊢ Iff (Eq m₁ (HMul.hMul m₂ 1)) (Or (Exists fun m' => And (Eq w.toList.head? (O …
        -/
      · revert h
        /-
          case neg
          ι : Type u_1
          M : ι → Type u_2
          inst✝² : (i : ι) → Monoid (M i)
          inst✝¹ : DecidableEq ι
          inst✝ : (i : ι) → DecidableEq (M i)
          j : ι
          m₂ : M j
          w : Monoid.CoprodI.Word M
          m₁ : M j
          hw : Not (Membership.mem w.toList.tail ⟨j, m₁⟩)
          hm1 : Not (Eq m₁ 1)
          ⊢ Not (Exists fun h => Eq (w.toList.head h).fst j) → Iff (Eq m₁ (HMul.hMul m₂  …
        -/
        rw [fstIdx]
        /-
          case neg
          ι : Type u_1
          M : ι → Type u_2
          inst✝² : (i : ι) → Monoid (M i)
          inst✝¹ : DecidableEq ι
          inst✝ : (i : ι) → DecidableEq (M i)
          j : ι
          m₂ : M j
          w : Monoid.CoprodI.Word M
          m₁ : M j
          hw : Not (Membership.mem w.toList.tail ⟨j, m₁⟩)
          hm1 : Not (Eq m₁ 1)
          ⊢ Not (Exists fun h => Eq (w.toList.head h).fst j) → Iff (Eq m₁ (HMul.hMul m₂  …
        -/
        cases w.toList
          /-
            case neg.nil
            ι : Type u_1
            M : ι → Type u_2
            inst✝² : (i : ι) → Monoid (M i)
            inst✝¹ : DecidableEq ι
            inst✝ : (i : ι) → DecidableEq (M i)
            j : ι
            m₂ : M j
            w : Monoid.CoprodI.Word M
            m₁ : M j
            hw : Not (Membership.mem w.toList.tail ⟨j, m₁⟩)
            hm1 : Not (Eq m₁ 1)
            ⊢ Not (Exists fun h => Eq (List.nil.head h).fst j) → Iff (Eq m₁ (HMul.hMul m₂  …
          -/
        · simp
          /-
            🎉 no goals
          -/
          /-
            case neg.cons
            ι : Type u_1
            M : ι → Type u_2
            inst✝² : (i : ι) → Monoid (M i)
            inst✝¹ : DecidableEq ι
            inst✝ : (i : ι) → DecidableEq (M i)
            j : ι
            m₂ : M j
            w : Monoid.CoprodI.Word M
            m₁ : M j
            hw : Not (Membership.mem w.toList.tail ⟨j, m₁⟩)
            hm1 : Not (Eq m₁ 1)
            head✝ : Sigma fun i => M i
            tail✝ : List (Sigma fun i => M i)
            ⊢ Not (Exists fun h => Eq ((List.cons head✝ tail✝).head h).fst j) → Iff (Eq m₁ …
          -/
        · simp (config := {contextual := true}) [Sigma.ext_iff]
          /-
            🎉 no goals
          -/
    /-
      case neg
      ι : Type u_1
      M : ι → Type u_2
      inst✝² : (i : ι) → Monoid (M i)
      inst✝¹ : DecidableEq ι
      inst✝ : (i : ι) → DecidableEq (M i)
      i j : ι
      m₁ : M i
      m₂ : M j
      w : Monoid.CoprodI.Word M
      hij : Not (Eq i j)
      ⊢ Iff
          (Or (Membership.mem w.toList.tail ⟨i, m₁⟩)
            (Or (And (Ne i j) (Exists fun h => Eq (w.toList.head h) ⟨i, m₁⟩))
              (And (Ne m₁ 1)
                (Exists fun h =>
                  Eq m₁
                    (Eq.rec
                      (let __src := (Monoid.CoprodI.Word.equivPair j) w;
                        { head := HMul.hMul m₂ (dite (Exists fun h => Eq (w.toList.h …
                      h)))))
          (Or (And (Not (Eq i j)) (Membership.mem w.toList ⟨i, m₁⟩)) (And (Ne m₁ 1)  …
    -/
  · rcases w with ⟨_ | _, _, _⟩ <;>
    /-
      case neg.mk.nil
      ι : Type u_1
      M : ι → Type u_2
      inst✝² : (i : ι) → Monoid (M i)
      inst✝¹ : DecidableEq ι
      inst✝ : (i : ι) → DecidableEq (M i)
      i j : ι
      m₁ : M i
      m₂ : M j
      hij : Not (Eq i j)
      ne_one✝ : ∀ (l : Sigma fun i => M i), Membership.mem List.nil l → Ne l.snd 1
      chain_ne✝ : List.Chain' (fun l l' => Ne l.fst l'.fst) List.nil
      ⊢ Iff
          (Or (Membership.mem { toList := List.nil, ne_one := ne_one✝, chain_ne := c …
            (Or (And (Ne i j) (Exists fun h => Eq ({ toList := List.nil, ne_one := n …
              (And (Ne m₁ 1)
                (Exists fun h =>
                  Eq m₁
                    (Eq.rec
                      (let __src := (Monoid.CoprodI.Word.equivPair j) { toList := Li …
                        { head := HMul.hMul m₂ (dite (Exists fun h => Eq ({ toList : …
                      h)))))
          (Or (And (Not (Eq i j)) (Membership.mem { toList := List.nil, ne_one := ne …
    -/
    /-
      🎉 no goals
    -/
    simp [or_comm, hij, Ne.symm hij]; rw [eq_comm]
                                      /-
                                        🎉 no goals
                                      -/


theorem mem_smul_iff_of_ne {i j : ι} (hij : i ≠ j) {m₁ : M i} {m₂ : M j} {w : Word M} :
    ⟨_, m₁⟩ ∈ (of m₂ • w).toList ↔ ⟨i, m₁⟩ ∈ w.toList := by
  /-
    ι : Type u_1
    M : ι → Type u_2
    inst✝² : (i : ι) → Monoid (M i)
    inst✝¹ : DecidableEq ι
    inst✝ : (i : ι) → DecidableEq (M i)
    i j : ι
    hij : Ne i j
    m₁ : M i
    m₂ : M j
    w : Monoid.CoprodI.Word M
    ⊢ Iff (Membership.mem (HSMul.hSMul (Monoid.CoprodI.of m₂) w).toList ⟨i, m₁⟩) ( …
  -/
  simp [mem_smul_iff, *]
  /-
    🎉 no goals
  -/


theorem cons_eq_smul {i} {m : M i} {ls h1 h2} :
    cons m ls h1 h2 = of m • ls := by
  /-
    ι : Type u_1
    M : ι → Type u_2
    inst✝² : (i : ι) → Monoid (M i)
    inst✝¹ : DecidableEq ι
    inst✝ : (i : ι) → DecidableEq (M i)
    i : ι
    m : M i
    ls : Monoid.CoprodI.Word M
    h1 : Ne ls.fstIdx (Option.some i)
    h2 : Ne m 1
    ⊢ Eq (Monoid.CoprodI.Word.cons m ls h1 h2) (HSMul.hSMul (Monoid.CoprodI.of m)  …
  -/
  rw [of_smul_def, equivPair_eq_of_fstIdx_ne _]
    /-
      ι : Type u_1
      M : ι → Type u_2
      inst✝² : (i : ι) → Monoid (M i)
      inst✝¹ : DecidableEq ι
      inst✝ : (i : ι) → DecidableEq (M i)
      i : ι
      m : M i
      ls : Monoid.CoprodI.Word M
      h1 : Ne ls.fstIdx (Option.some i)
      h2 : Ne m 1
      ⊢ Eq (Monoid.CoprodI.Word.cons m ls h1 h2)
          (Monoid.CoprodI.Word.rcons
            (let __src := { head := 1, tail := ls, fstIdx_ne := ?m.185687 };
            { head := HMul.hMul m { head := 1, tail := ls, fstIdx_ne := ?m.185687 }. …
    -/
  · simp [cons, rcons, h2]
    /-
      🎉 no goals
    -/
    /-
      ι : Type u_1
      M : ι → Type u_2
      inst✝² : (i : ι) → Monoid (M i)
      inst✝¹ : DecidableEq ι
      inst✝ : (i : ι) → DecidableEq (M i)
      i : ι
      m : M i
      ls : Monoid.CoprodI.Word M
      h1 : Ne ls.fstIdx (Option.some i)
      h2 : Ne m 1
      ⊢ Ne ls.fstIdx (Option.some i)
    -/
  · exact h1
    /-
      🎉 no goals
    -/


theorem rcons_eq_smul {i} (p : Pair M i) :
    rcons p = of p.head • p.tail := by
  /-
    ι : Type u_1
    M : ι → Type u_2
    inst✝² : (i : ι) → Monoid (M i)
    inst✝¹ : DecidableEq ι
    inst✝ : (i : ι) → DecidableEq (M i)
    i : ι
    p : Monoid.CoprodI.Word.Pair M i
    ⊢ Eq (Monoid.CoprodI.Word.rcons p) (HSMul.hSMul (Monoid.CoprodI.of p.head) p.t …
  -/
  simp [of_smul_def]
  /-
    🎉 no goals
  -/


@[simp]
theorem equivPair_head_smul_equivPair_tail {i : ι} (w : Word M) :
    of (equivPair i w).head • (equivPair i w).tail = w := by
  /-
    ι : Type u_1
    M : ι → Type u_2
    inst✝² : (i : ι) → Monoid (M i)
    inst✝¹ : DecidableEq ι
    inst✝ : (i : ι) → DecidableEq (M i)
    i : ι
    w : Monoid.CoprodI.Word M
    ⊢ Eq (HSMul.hSMul (Monoid.CoprodI.of ((Monoid.CoprodI.Word.equivPair i) w).hea …
  -/
  rw [← rcons_eq_smul, ← equivPair_symm, Equiv.symm_apply_apply]
  /-
    🎉 no goals
  -/


theorem equivPair_tail_eq_inv_smul {G : ι → Type*} [∀ i, Group (G i)]
    [∀i, DecidableEq (G i)] {i} (w : Word G) :
    (equivPair i w).tail = (of (equivPair i w).head)⁻¹ • w :=
  Eq.symm <| inv_smul_eq_iff.2 (equivPair_head_smul_equivPair_tail w).symm


theorem smul_induction {C : Word M → Prop} (h_empty : C empty)
    (h_smul : ∀ (i) (m : M i) (w), C w → C (of m • w)) (w : Word M) : C w := by
  induction w using consRecOn with
  | h_empty => exact h_empty
  | h_cons _ _ _ _ _ ih =>
    rw [cons_eq_smul]
    exact h_smul _ _ _ ih


@[simp]
theorem prod_smul (m) : ∀ w : Word M, prod (m • w) = m * prod w := by
  induction m using CoprodI.induction_on with
  | h_one =>
    intro
    rw [one_smul, one_mul]
  | h_of _ =>
    intros
    rw [of_smul_def, prod_rcons, of.map_mul, mul_assoc, ← prod_rcons, ← equivPair_symm,
      Equiv.symm_apply_apply]
  | h_mul x y hx hy =>
    intro w
    rw [mul_smul, hx, hy, mul_assoc]


/-- Each element of the free product corresponds to a unique reduced word. -/
def equiv : CoprodI M ≃ Word M where
  toFun m := m • empty
  invFun w := prod w
                   /-
                     ι : Type u_1
                     M : ι → Type u_2
                     inst✝³ : (i : ι) → Monoid (M i)
                     N : Type u_3
                     inst✝² : Monoid N
                     inst✝¹ : DecidableEq ι
                     inst✝ : (i : ι) → DecidableEq (M i)
                     m : Monoid.CoprodI M
                     ⊢ Eq ((fun w => w.prod) ((fun m => HSMul.hSMul m Monoid.CoprodI.Word.empty) m) …
                   -/
  left_inv m := by dsimp only; rw [prod_smul, prod_empty, mul_one]
                               /-
                                 🎉 no goals
                               -/
  right_inv := by
    /-
      ι : Type u_1
      M : ι → Type u_2
      inst✝³ : (i : ι) → Monoid (M i)
      N : Type u_3
      inst✝² : Monoid N
      inst✝¹ : DecidableEq ι
      inst✝ : (i : ι) → DecidableEq (M i)
      ⊢ Function.RightInverse (fun w => w.prod) fun m => HSMul.hSMul m Monoid.Coprod …
    -/
    apply smul_induction
      /-
        case h_empty
        ι : Type u_1
        M : ι → Type u_2
        inst✝³ : (i : ι) → Monoid (M i)
        N : Type u_3
        inst✝² : Monoid N
        inst✝¹ : DecidableEq ι
        inst✝ : (i : ι) → DecidableEq (M i)
        ⊢ Eq ((fun m => HSMul.hSMul m Monoid.CoprodI.Word.empty) ((fun w => w.prod) Mo …
      -/
    · dsimp only
      /-
        case h_empty
        ι : Type u_1
        M : ι → Type u_2
        inst✝³ : (i : ι) → Monoid (M i)
        N : Type u_3
        inst✝² : Monoid N
        inst✝¹ : DecidableEq ι
        inst✝ : (i : ι) → DecidableEq (M i)
        ⊢ Eq (HSMul.hSMul Monoid.CoprodI.Word.empty.prod Monoid.CoprodI.Word.empty) Mo …
      -/
      rw [prod_empty, one_smul]
      /-
        🎉 no goals
      -/
      /-
        case h_smul
        ι : Type u_1
        M : ι → Type u_2
        inst✝³ : (i : ι) → Monoid (M i)
        N : Type u_3
        inst✝² : Monoid N
        inst✝¹ : DecidableEq ι
        inst✝ : (i : ι) → DecidableEq (M i)
        ⊢ ∀ (i : ι) (m : M i) (w : Monoid.CoprodI.Word M), Eq ((fun m => HSMul.hSMul m …
      -/
    · dsimp only
      /-
        case h_smul
        ι : Type u_1
        M : ι → Type u_2
        inst✝³ : (i : ι) → Monoid (M i)
        N : Type u_3
        inst✝² : Monoid N
        inst✝¹ : DecidableEq ι
        inst✝ : (i : ι) → DecidableEq (M i)
        ⊢ ∀ (i : ι) (m : M i) (w : Monoid.CoprodI.Word M), Eq (HSMul.hSMul w.prod Mono …
      -/
      intro i m w ih
      /-
        case h_smul
        ι : Type u_1
        M : ι → Type u_2
        inst✝³ : (i : ι) → Monoid (M i)
        N : Type u_3
        inst✝² : Monoid N
        inst✝¹ : DecidableEq ι
        inst✝ : (i : ι) → DecidableEq (M i)
        i : ι
        m : M i
        w : Monoid.CoprodI.Word M
        ih : Eq (HSMul.hSMul w.prod Monoid.CoprodI.Word.empty) w
        ⊢ Eq (HSMul.hSMul (HSMul.hSMul (Monoid.CoprodI.of m) w).prod Monoid.CoprodI.Wo …
      -/
      rw [prod_smul, mul_smul, ih]
      /-
        🎉 no goals
      -/


instance : DecidableEq (Word M) :=
  Function.Injective.decidableEq fun _ _ => Word.ext


instance : DecidableEq (CoprodI M) :=
  Equiv.decidableEq Word.equiv


/-- A `NeWord M i j` is a representation of a non-empty reduced words where the first letter comes
from `M i` and the last letter comes from `M j`. It can be constructed from singletons and via
concatenation, and thus provides a useful induction principle. -/
--@[nolint has_nonempty_instance] Porting note (https://github.com/leanprover-community/mathlib4/issues/5171): commented out
inductive NeWord : ι → ι → Type _
  | singleton : ∀ {i : ι} (x : M i), x ≠ 1 → NeWord i i
  | append : ∀ {i j k l} (_w₁ : NeWord i j) (_hne : j ≠ k) (_w₂ : NeWord k l), NeWord i l


/-- The list represented by a given `NeWord` -/
@[simp]
def toList : ∀ {i j} (_w : NeWord M i j), List (Σi, M i)
  | i, _, singleton x _ => [⟨i, x⟩]
  | _, _, append w₁ _ w₂ => w₁.toList ++ w₂.toList


theorem toList_ne_nil {i j} (w : NeWord M i j) : w.toList ≠ List.nil := by
  /-
    ι : Type u_1
    M : ι → Type u_2
    inst✝ : (i : ι) → Monoid (M i)
    i j : ι
    w : Monoid.CoprodI.NeWord M i j
    ⊢ Ne w.toList List.nil
  -/
  induction w
    /-
      case singleton
      ι : Type u_1
      M : ι → Type u_2
      inst✝ : (i : ι) → Monoid (M i)
      i j i✝ : ι
      x✝ : M i✝
      a✝ : Ne x✝ 1
      ⊢ Ne (Monoid.CoprodI.NeWord.singleton x✝ a✝).toList List.nil
    -/
  · rintro ⟨rfl⟩
    /-
      🎉 no goals
    -/
    /-
      case append
      ι : Type u_1
      M : ι → Type u_2
      inst✝ : (i : ι) → Monoid (M i)
      i j i✝ j✝ k✝ l✝ : ι
      _w₁✝ : Monoid.CoprodI.NeWord M i✝ j✝
      _hne✝ : Ne j✝ k✝
      _w₂✝ : Monoid.CoprodI.NeWord M k✝ l✝
      _w₁_ih✝ : Ne _w₁✝.toList List.nil
      _w₂_ih✝ : Ne _w₂✝.toList List.nil
      ⊢ Ne (_w₁✝.append _hne✝ _w₂✝).toList List.nil
    -/
  · apply List.append_ne_nil_of_left_ne_nil
    /-
      case append.h
      ι : Type u_1
      M : ι → Type u_2
      inst✝ : (i : ι) → Monoid (M i)
      i j i✝ j✝ k✝ l✝ : ι
      _w₁✝ : Monoid.CoprodI.NeWord M i✝ j✝
      _hne✝ : Ne j✝ k✝
      _w₂✝ : Monoid.CoprodI.NeWord M k✝ l✝
      _w₁_ih✝ : Ne _w₁✝.toList List.nil
      _w₂_ih✝ : Ne _w₂✝.toList List.nil
      ⊢ Ne _w₁✝.toList List.nil
    -/
    assumption
    /-
      🎉 no goals
    -/


/-- The first letter of a `NeWord` -/
@[simp]
def head : ∀ {i j} (_w : NeWord M i j), M i
  | _, _, singleton x _ => x
  | _, _, append w₁ _ _ => w₁.head


/-- The last letter of a `NeWord` -/
@[simp]
def last : ∀ {i j} (_w : NeWord M i j), M j
  | _, _, singleton x _hne1 => x
  | _, _, append _w₁ _hne w₂ => w₂.last


@[simp]
theorem toList_head? {i j} (w : NeWord M i j) : w.toList.head? = Option.some ⟨i, w.head⟩ := by
  /-
    ι : Type u_1
    M : ι → Type u_2
    inst✝ : (i : ι) → Monoid (M i)
    i j : ι
    w : Monoid.CoprodI.NeWord M i j
    ⊢ Eq w.toList.head? (Option.some ⟨i, w.head⟩)
  -/
  rw [← Option.mem_def]
  /-
    ι : Type u_1
    M : ι → Type u_2
    inst✝ : (i : ι) → Monoid (M i)
    i j : ι
    w : Monoid.CoprodI.NeWord M i j
    ⊢ Membership.mem w.toList.head? ⟨i, w.head⟩
  -/
  induction w
    /-
      case singleton
      ι : Type u_1
      M : ι → Type u_2
      inst✝ : (i : ι) → Monoid (M i)
      i j i✝ : ι
      x✝ : M i✝
      a✝ : Ne x✝ 1
      ⊢ Membership.mem (Monoid.CoprodI.NeWord.singleton x✝ a✝).toList.head? ⟨i✝, (Mo …
    -/
  · rw [Option.mem_def]
    /-
      case singleton
      ι : Type u_1
      M : ι → Type u_2
      inst✝ : (i : ι) → Monoid (M i)
      i j i✝ : ι
      x✝ : M i✝
      a✝ : Ne x✝ 1
      ⊢ Eq (Monoid.CoprodI.NeWord.singleton x✝ a✝).toList.head? (Option.some ⟨i✝, (M …
    -/
    rfl
    /-
      🎉 no goals
    -/
    /-
      case append
      ι : Type u_1
      M : ι → Type u_2
      inst✝ : (i : ι) → Monoid (M i)
      i j i✝ j✝ k✝ l✝ : ι
      _w₁✝ : Monoid.CoprodI.NeWord M i✝ j✝
      _hne✝ : Ne j✝ k✝
      _w₂✝ : Monoid.CoprodI.NeWord M k✝ l✝
      _w₁_ih✝ : Membership.mem _w₁✝.toList.head? ⟨i✝, _w₁✝.head⟩
      _w₂_ih✝ : Membership.mem _w₂✝.toList.head? ⟨k✝, _w₂✝.head⟩
      ⊢ Membership.mem (_w₁✝.append _hne✝ _w₂✝).toList.head? ⟨i✝, (_w₁✝.append _hne✝ …
    -/
  · exact List.mem_head?_append_of_mem_head? (by assumption)
    /-
      🎉 no goals
    -/


@[simp]
theorem toList_getLast? {i j} (w : NeWord M i j) : w.toList.getLast? = Option.some ⟨j, w.last⟩ := by
  /-
    ι : Type u_1
    M : ι → Type u_2
    inst✝ : (i : ι) → Monoid (M i)
    i j : ι
    w : Monoid.CoprodI.NeWord M i j
    ⊢ Eq w.toList.getLast? (Option.some ⟨j, w.last⟩)
  -/
  rw [← Option.mem_def]
  /-
    ι : Type u_1
    M : ι → Type u_2
    inst✝ : (i : ι) → Monoid (M i)
    i j : ι
    w : Monoid.CoprodI.NeWord M i j
    ⊢ Membership.mem w.toList.getLast? ⟨j, w.last⟩
  -/
  induction w
    /-
      case singleton
      ι : Type u_1
      M : ι → Type u_2
      inst✝ : (i : ι) → Monoid (M i)
      i j i✝ : ι
      x✝ : M i✝
      a✝ : Ne x✝ 1
      ⊢ Membership.mem (Monoid.CoprodI.NeWord.singleton x✝ a✝).toList.getLast? ⟨i✝,  …
    -/
  · rw [Option.mem_def]
    /-
      case singleton
      ι : Type u_1
      M : ι → Type u_2
      inst✝ : (i : ι) → Monoid (M i)
      i j i✝ : ι
      x✝ : M i✝
      a✝ : Ne x✝ 1
      ⊢ Eq (Monoid.CoprodI.NeWord.singleton x✝ a✝).toList.getLast? (Option.some ⟨i✝, …
    -/
    rfl
    /-
      🎉 no goals
    -/
    /-
      case append
      ι : Type u_1
      M : ι → Type u_2
      inst✝ : (i : ι) → Monoid (M i)
      i j i✝ j✝ k✝ l✝ : ι
      _w₁✝ : Monoid.CoprodI.NeWord M i✝ j✝
      _hne✝ : Ne j✝ k✝
      _w₂✝ : Monoid.CoprodI.NeWord M k✝ l✝
      _w₁_ih✝ : Membership.mem _w₁✝.toList.getLast? ⟨j✝, _w₁✝.last⟩
      _w₂_ih✝ : Membership.mem _w₂✝.toList.getLast? ⟨l✝, _w₂✝.last⟩
      ⊢ Membership.mem (_w₁✝.append _hne✝ _w₂✝).toList.getLast? ⟨l✝, (_w₁✝.append _h …
    -/
  · exact List.mem_getLast?_append_of_mem_getLast? (by assumption)
    /-
      🎉 no goals
    -/


/-- The `Word M` represented by a `NeWord M i j` -/
def toWord {i j} (w : NeWord M i j) : Word M where
  toList := w.toList
  ne_one := by
    /-
      ι : Type u_1
      M : ι → Type u_2
      inst✝¹ : (i : ι) → Monoid (M i)
      N : Type u_3
      inst✝ : Monoid N
      i j : ι
      w : Monoid.CoprodI.NeWord M i j
      ⊢ ∀ (l : Sigma fun i => M i), Membership.mem w.toList l → Ne l.snd 1
    -/
    induction w
      /-
        case singleton
        ι : Type u_1
        M : ι → Type u_2
        inst✝¹ : (i : ι) → Monoid (M i)
        N : Type u_3
        inst✝ : Monoid N
        i j i✝ : ι
        x✝ : M i✝
        a✝ : Ne x✝ 1
        ⊢ ∀ (l : Sigma fun i => M i), Membership.mem (Monoid.CoprodI.NeWord.singleton  …
      -/
    · simpa only [toList, List.mem_singleton, ne_eq, forall_eq]
      /-
        🎉 no goals
      -/
      /-
        case append
        ι : Type u_1
        M : ι → Type u_2
        inst✝¹ : (i : ι) → Monoid (M i)
        N : Type u_3
        inst✝ : Monoid N
        i j i✝ j✝ k✝ l✝ : ι
        _w₁✝ : Monoid.CoprodI.NeWord M i✝ j✝
        _hne✝ : Ne j✝ k✝
        _w₂✝ : Monoid.CoprodI.NeWord M k✝ l✝
        _w₁_ih✝ : ∀ (l : Sigma fun i => M i), Membership.mem _w₁✝.toList l → Ne l.snd 1
        _w₂_ih✝ : ∀ (l : Sigma fun i => M i), Membership.mem _w₂✝.toList l → Ne l.snd 1
        ⊢ ∀ (l : Sigma fun i => M i), Membership.mem (_w₁✝.append _hne✝ _w₂✝).toList l …
      -/
    · intro l h
      /-
        case append
        ι : Type u_1
        M : ι → Type u_2
        inst✝¹ : (i : ι) → Monoid (M i)
        N : Type u_3
        inst✝ : Monoid N
        i j i✝ j✝ k✝ l✝ : ι
        _w₁✝ : Monoid.CoprodI.NeWord M i✝ j✝
        _hne✝ : Ne j✝ k✝
        _w₂✝ : Monoid.CoprodI.NeWord M k✝ l✝
        _w₁_ih✝ : ∀ (l : Sigma fun i => M i), Membership.mem _w₁✝.toList l → Ne l.snd 1
        _w₂_ih✝ : ∀ (l : Sigma fun i => M i), Membership.mem _w₂✝.toList l → Ne l.snd 1
        l : Sigma fun i => M i
        h : Membership.mem (_w₁✝.append _hne✝ _w₂✝).toList l
        ⊢ Ne l.snd 1
      -/
      simp only [toList, List.mem_append] at h
      /-
        case append
        ι : Type u_1
        M : ι → Type u_2
        inst✝¹ : (i : ι) → Monoid (M i)
        N : Type u_3
        inst✝ : Monoid N
        i j i✝ j✝ k✝ l✝ : ι
        _w₁✝ : Monoid.CoprodI.NeWord M i✝ j✝
        _hne✝ : Ne j✝ k✝
        _w₂✝ : Monoid.CoprodI.NeWord M k✝ l✝
        _w₁_ih✝ : ∀ (l : Sigma fun i => M i), Membership.mem _w₁✝.toList l → Ne l.snd 1
        _w₂_ih✝ : ∀ (l : Sigma fun i => M i), Membership.mem _w₂✝.toList l → Ne l.snd 1
        l : Sigma fun i => M i
        h : Or (Membership.mem _w₁✝.toList l) (Membership.mem _w₂✝.toList l)
        ⊢ Ne l.snd 1
      -/
                  /-
                    🎉 no goals
                  -/
      cases h <;> aesop
                  /-
                    🎉 no goals
                  -/
  chain_ne := by
    /-
      ι : Type u_1
      M : ι → Type u_2
      inst✝¹ : (i : ι) → Monoid (M i)
      N : Type u_3
      inst✝ : Monoid N
      i j : ι
      w : Monoid.CoprodI.NeWord M i j
      ⊢ List.Chain' (fun l l' => Ne l.fst l'.fst) w.toList
    -/
    induction w
      /-
        case singleton
        ι : Type u_1
        M : ι → Type u_2
        inst✝¹ : (i : ι) → Monoid (M i)
        N : Type u_3
        inst✝ : Monoid N
        i j i✝ : ι
        x✝ : M i✝
        a✝ : Ne x✝ 1
        ⊢ List.Chain' (fun l l' => Ne l.fst l'.fst) (Monoid.CoprodI.NeWord.singleton x …
      -/
    · exact List.chain'_singleton _
      /-
        🎉 no goals
      -/
      /-
        case append
        ι : Type u_1
        M : ι → Type u_2
        inst✝¹ : (i : ι) → Monoid (M i)
        N : Type u_3
        inst✝ : Monoid N
        i j i✝ j✝ k✝ l✝ : ι
        _w₁✝ : Monoid.CoprodI.NeWord M i✝ j✝
        _hne✝ : Ne j✝ k✝
        _w₂✝ : Monoid.CoprodI.NeWord M k✝ l✝
        _w₁_ih✝ : List.Chain' (fun l l' => Ne l.fst l'.fst) _w₁✝.toList
        _w₂_ih✝ : List.Chain' (fun l l' => Ne l.fst l'.fst) _w₂✝.toList
        ⊢ List.Chain' (fun l l' => Ne l.fst l'.fst) (_w₁✝.append _hne✝ _w₂✝).toList
      -/
    · refine List.Chain'.append (by assumption) (by assumption) ?_
      /-
        case append
        ι : Type u_1
        M : ι → Type u_2
        inst✝¹ : (i : ι) → Monoid (M i)
        N : Type u_3
        inst✝ : Monoid N
        i j i✝ j✝ k✝ l✝ : ι
        _w₁✝ : Monoid.CoprodI.NeWord M i✝ j✝
        _hne✝ : Ne j✝ k✝
        _w₂✝ : Monoid.CoprodI.NeWord M k✝ l✝
        _w₁_ih✝ : List.Chain' (fun l l' => Ne l.fst l'.fst) _w₁✝.toList
        _w₂_ih✝ : List.Chain' (fun l l' => Ne l.fst l'.fst) _w₂✝.toList
        ⊢ ∀ (x : Sigma fun i => M i), Membership.mem _w₁✝.toList.getLast? x → ∀ (y : S …
      -/
      intro x hx y hy
      /-
        case append
        ι : Type u_1
        M : ι → Type u_2
        inst✝¹ : (i : ι) → Monoid (M i)
        N : Type u_3
        inst✝ : Monoid N
        i j i✝ j✝ k✝ l✝ : ι
        _w₁✝ : Monoid.CoprodI.NeWord M i✝ j✝
        _hne✝ : Ne j✝ k✝
        _w₂✝ : Monoid.CoprodI.NeWord M k✝ l✝
        _w₁_ih✝ : List.Chain' (fun l l' => Ne l.fst l'.fst) _w₁✝.toList
        _w₂_ih✝ : List.Chain' (fun l l' => Ne l.fst l'.fst) _w₂✝.toList
        x : Sigma fun i => M i
        hx : Membership.mem _w₁✝.toList.getLast? x
        y : Sigma fun i => M i
        hy : Membership.mem _w₂✝.toList.head? y
        ⊢ Ne x.fst y.fst
      -/
      rw [toList_getLast?, Option.mem_some_iff] at hx
      /-
        case append
        ι : Type u_1
        M : ι → Type u_2
        inst✝¹ : (i : ι) → Monoid (M i)
        N : Type u_3
        inst✝ : Monoid N
        i j i✝ j✝ k✝ l✝ : ι
        _w₁✝ : Monoid.CoprodI.NeWord M i✝ j✝
        _hne✝ : Ne j✝ k✝
        _w₂✝ : Monoid.CoprodI.NeWord M k✝ l✝
        _w₁_ih✝ : List.Chain' (fun l l' => Ne l.fst l'.fst) _w₁✝.toList
        _w₂_ih✝ : List.Chain' (fun l l' => Ne l.fst l'.fst) _w₂✝.toList
        x : Sigma fun i => M i
        hx : Eq ⟨j✝, _w₁✝.last⟩ x
        y : Sigma fun i => M i
        hy : Membership.mem _w₂✝.toList.head? y
        ⊢ Ne x.fst y.fst
      -/
      rw [toList_head?, Option.mem_some_iff] at hy
      /-
        case append
        ι : Type u_1
        M : ι → Type u_2
        inst✝¹ : (i : ι) → Monoid (M i)
        N : Type u_3
        inst✝ : Monoid N
        i j i✝ j✝ k✝ l✝ : ι
        _w₁✝ : Monoid.CoprodI.NeWord M i✝ j✝
        _hne✝ : Ne j✝ k✝
        _w₂✝ : Monoid.CoprodI.NeWord M k✝ l✝
        _w₁_ih✝ : List.Chain' (fun l l' => Ne l.fst l'.fst) _w₁✝.toList
        _w₂_ih✝ : List.Chain' (fun l l' => Ne l.fst l'.fst) _w₂✝.toList
        x : Sigma fun i => M i
        hx : Eq ⟨j✝, _w₁✝.last⟩ x
        y : Sigma fun i => M i
        hy : Eq ⟨k✝, _w₂✝.head⟩ y
        ⊢ Ne x.fst y.fst
      -/
      subst hx
      /-
        case append
        ι : Type u_1
        M : ι → Type u_2
        inst✝¹ : (i : ι) → Monoid (M i)
        N : Type u_3
        inst✝ : Monoid N
        i j i✝ j✝ k✝ l✝ : ι
        _w₁✝ : Monoid.CoprodI.NeWord M i✝ j✝
        _hne✝ : Ne j✝ k✝
        _w₂✝ : Monoid.CoprodI.NeWord M k✝ l✝
        _w₁_ih✝ : List.Chain' (fun l l' => Ne l.fst l'.fst) _w₁✝.toList
        _w₂_ih✝ : List.Chain' (fun l l' => Ne l.fst l'.fst) _w₂✝.toList
        y : Sigma fun i => M i
        hy : Eq ⟨k✝, _w₂✝.head⟩ y
        ⊢ Ne ⟨j✝, _w₁✝.last⟩.fst y.fst
      -/
      subst hy
      /-
        case append
        ι : Type u_1
        M : ι → Type u_2
        inst✝¹ : (i : ι) → Monoid (M i)
        N : Type u_3
        inst✝ : Monoid N
        i j i✝ j✝ k✝ l✝ : ι
        _w₁✝ : Monoid.CoprodI.NeWord M i✝ j✝
        _hne✝ : Ne j✝ k✝
        _w₂✝ : Monoid.CoprodI.NeWord M k✝ l✝
        _w₁_ih✝ : List.Chain' (fun l l' => Ne l.fst l'.fst) _w₁✝.toList
        _w₂_ih✝ : List.Chain' (fun l l' => Ne l.fst l'.fst) _w₂✝.toList
        ⊢ Ne ⟨j✝, _w₁✝.last⟩.fst ⟨k✝, _w₂✝.head⟩.fst
      -/
      assumption
      /-
        🎉 no goals
      -/


/-- Every nonempty `Word M` can be constructed as a `NeWord M i j` -/
theorem of_word (w : Word M) (h : w ≠ empty) : ∃ (i j : _) (w' : NeWord M i j), w'.toWord = w := by
  suffices ∃ (i j : _) (w' : NeWord M i j), w'.toWord.toList = w.toList by
    rcases this with ⟨i, j, w, h⟩
    refine ⟨i, j, w, ?_⟩
    ext
    rw [h]
  /-
    ι : Type u_1
    M : ι → Type u_2
    inst✝ : (i : ι) → Monoid (M i)
    w : Monoid.CoprodI.Word M
    h : Ne w Monoid.CoprodI.Word.empty
    ⊢ Exists fun i => Exists fun j => Exists fun w' => Eq w'.toWord.toList w.toList
  -/
  cases' w with l hnot1 hchain
  /-
    case mk
    ι : Type u_1
    M : ι → Type u_2
    inst✝ : (i : ι) → Monoid (M i)
    l : List (Sigma fun i => M i)
    hnot1 : ∀ (l_1 : Sigma fun i => M i), Membership.mem l l_1 → Ne l_1.snd 1
    hchain : List.Chain' (fun l l' => Ne l.fst l'.fst) l
    h : Ne { toList := l, ne_one := hnot1, chain_ne := hchain } Monoid.CoprodI.Wor …
    ⊢ Exists fun i => Exists fun j => Exists fun w' => Eq w'.toWord.toList { toLis …
  -/
  induction' l with x l hi
    /-
      case mk.nil
      ι : Type u_1
      M : ι → Type u_2
      inst✝ : (i : ι) → Monoid (M i)
      hnot1 : ∀ (l : Sigma fun i => M i), Membership.mem List.nil l → Ne l.snd 1
      hchain : List.Chain' (fun l l' => Ne l.fst l'.fst) List.nil
      h : Ne { toList := List.nil, ne_one := hnot1, chain_ne := hchain } Monoid.Copr …
      ⊢ Exists fun i => Exists fun j => Exists fun w' => Eq w'.toWord.toList { toLis …
    -/
  · contradiction
    /-
      🎉 no goals
    -/
    /-
      case mk.cons
      ι : Type u_1
      M : ι → Type u_2
      inst✝ : (i : ι) → Monoid (M i)
      x : Sigma fun i => M i
      l : List (Sigma fun i => M i)
      hi : ∀ (hnot1 : ∀ (l_1 : Sigma fun i => M i), Membership.mem l l_1 → Ne l_1.sn …
      hnot1 : ∀ (l_1 : Sigma fun i => M i), Membership.mem (List.cons x l) l_1 → Ne  …
      hchain : List.Chain' (fun l l' => Ne l.fst l'.fst) (List.cons x l)
      h : Ne { toList := List.cons x l, ne_one := hnot1, chain_ne := hchain } Monoid …
      ⊢ Exists fun i => Exists fun j => Exists fun w' => Eq w'.toWord.toList { toLis …
    -/
  · rw [List.forall_mem_cons] at hnot1
    /-
      case mk.cons
      ι : Type u_1
      M : ι → Type u_2
      inst✝ : (i : ι) → Monoid (M i)
      x : Sigma fun i => M i
      l : List (Sigma fun i => M i)
      hi : ∀ (hnot1 : ∀ (l_1 : Sigma fun i => M i), Membership.mem l l_1 → Ne l_1.sn …
      hnot1✝ : ∀ (l_1 : Sigma fun i => M i), Membership.mem (List.cons x l) l_1 → Ne …
      hnot1 : And (Ne x.snd 1) (∀ (x : Sigma fun i => M i), Membership.mem l x → Ne  …
      hchain : List.Chain' (fun l l' => Ne l.fst l'.fst) (List.cons x l)
      h : Ne { toList := List.cons x l, ne_one := hnot1✝, chain_ne := hchain } Monoi …
      ⊢ Exists fun i => Exists fun j => Exists fun w' => Eq w'.toWord.toList { toLis …
    -/
    cases' l with y l
      /-
        case mk.cons.nil
        ι : Type u_1
        M : ι → Type u_2
        inst✝ : (i : ι) → Monoid (M i)
        x : Sigma fun i => M i
        hi : ∀ (hnot1 : ∀ (l : Sigma fun i => M i), Membership.mem List.nil l → Ne l.s …
        hnot1✝ : ∀ (l : Sigma fun i => M i), Membership.mem (List.cons x List.nil) l → …
        hnot1 : And (Ne x.snd 1) (∀ (x : Sigma fun i => M i), Membership.mem List.nil  …
        hchain : List.Chain' (fun l l' => Ne l.fst l'.fst) (List.cons x List.nil)
        h : Ne { toList := List.cons x List.nil, ne_one := hnot1✝, chain_ne := hchain  …
        ⊢ Exists fun i => Exists fun j => Exists fun w' => Eq w'.toWord.toList { toLis …
      -/
    · refine ⟨x.1, x.1, singleton x.2 hnot1.1, ?_⟩
      /-
        case mk.cons.nil
        ι : Type u_1
        M : ι → Type u_2
        inst✝ : (i : ι) → Monoid (M i)
        x : Sigma fun i => M i
        hi : ∀ (hnot1 : ∀ (l : Sigma fun i => M i), Membership.mem List.nil l → Ne l.s …
        hnot1✝ : ∀ (l : Sigma fun i => M i), Membership.mem (List.cons x List.nil) l → …
        hnot1 : And (Ne x.snd 1) (∀ (x : Sigma fun i => M i), Membership.mem List.nil  …
        hchain : List.Chain' (fun l l' => Ne l.fst l'.fst) (List.cons x List.nil)
        h : Ne { toList := List.cons x List.nil, ne_one := hnot1✝, chain_ne := hchain  …
        ⊢ Eq (Monoid.CoprodI.NeWord.singleton x.snd ⋯).toWord.toList { toList := List. …
      -/
      simp [toWord]
      /-
        🎉 no goals
      -/
      /-
        case mk.cons.cons
        ι : Type u_1
        M : ι → Type u_2
        inst✝ : (i : ι) → Monoid (M i)
        x y : Sigma fun i => M i
        l : List (Sigma fun i => M i)
        hi : ∀ (hnot1 : ∀ (l_1 : Sigma fun i => M i), Membership.mem (List.cons y l) l …
        hnot1✝ : ∀ (l_1 : Sigma fun i => M i), Membership.mem (List.cons x (List.cons  …
        hnot1 : And (Ne x.snd 1) (∀ (x : Sigma fun i => M i), Membership.mem (List.con …
        hchain : List.Chain' (fun l l' => Ne l.fst l'.fst) (List.cons x (List.cons y l))
        h : Ne { toList := List.cons x (List.cons y l), ne_one := hnot1✝, chain_ne :=  …
        ⊢ Exists fun i => Exists fun j => Exists fun w' => Eq w'.toWord.toList { toLis …
      -/
    · rw [List.chain'_cons] at hchain
      /-
        case mk.cons.cons
        ι : Type u_1
        M : ι → Type u_2
        inst✝ : (i : ι) → Monoid (M i)
        x y : Sigma fun i => M i
        l : List (Sigma fun i => M i)
        hi : ∀ (hnot1 : ∀ (l_1 : Sigma fun i => M i), Membership.mem (List.cons y l) l …
        hnot1✝ : ∀ (l_1 : Sigma fun i => M i), Membership.mem (List.cons x (List.cons  …
        hnot1 : And (Ne x.snd 1) (∀ (x : Sigma fun i => M i), Membership.mem (List.con …
        hchain✝ : List.Chain' (fun l l' => Ne l.fst l'.fst) (List.cons x (List.cons y  …
        hchain : And (Ne x.fst y.fst) (List.Chain' (fun l l' => Ne l.fst l'.fst) (List …
        h : Ne { toList := List.cons x (List.cons y l), ne_one := hnot1✝, chain_ne :=  …
        ⊢ Exists fun i => Exists fun j => Exists fun w' => Eq w'.toWord.toList { toLis …
      -/
      specialize hi hnot1.2 hchain.2 (by rintro ⟨rfl⟩)
      /-
        case mk.cons.cons
        ι : Type u_1
        M : ι → Type u_2
        inst✝ : (i : ι) → Monoid (M i)
        x y : Sigma fun i => M i
        l : List (Sigma fun i => M i)
        hnot1✝ : ∀ (l_1 : Sigma fun i => M i), Membership.mem (List.cons x (List.cons  …
        hnot1 : And (Ne x.snd 1) (∀ (x : Sigma fun i => M i), Membership.mem (List.con …
        hchain✝ : List.Chain' (fun l l' => Ne l.fst l'.fst) (List.cons x (List.cons y  …
        hchain : And (Ne x.fst y.fst) (List.Chain' (fun l l' => Ne l.fst l'.fst) (List …
        h : Ne { toList := List.cons x (List.cons y l), ne_one := hnot1✝, chain_ne :=  …
        hi : Exists fun i => Exists fun j => Exists fun w' => Eq w'.toWord.toList { to …
        ⊢ Exists fun i => Exists fun j => Exists fun w' => Eq w'.toWord.toList { toLis …
      -/
      obtain ⟨i, j, w', hw' : w'.toList = y::l⟩ := hi
      /-
        case mk.cons.cons.intro.intro.intro
        ι : Type u_1
        M : ι → Type u_2
        inst✝ : (i : ι) → Monoid (M i)
        x y : Sigma fun i => M i
        l : List (Sigma fun i => M i)
        hnot1✝ : ∀ (l_1 : Sigma fun i => M i), Membership.mem (List.cons x (List.cons  …
        hnot1 : And (Ne x.snd 1) (∀ (x : Sigma fun i => M i), Membership.mem (List.con …
        hchain✝ : List.Chain' (fun l l' => Ne l.fst l'.fst) (List.cons x (List.cons y  …
        hchain : And (Ne x.fst y.fst) (List.Chain' (fun l l' => Ne l.fst l'.fst) (List …
        h : Ne { toList := List.cons x (List.cons y l), ne_one := hnot1✝, chain_ne :=  …
        i j : ι
        w' : Monoid.CoprodI.NeWord M i j
        hw' : Eq w'.toList (List.cons y l)
        ⊢ Exists fun i => Exists fun j => Exists fun w' => Eq w'.toWord.toList { toLis …
      -/
      obtain rfl : y = ⟨i, w'.head⟩ := by simpa [hw'] using w'.toList_head?
      /-
        case mk.cons.cons.intro.intro.intro
        ι : Type u_1
        M : ι → Type u_2
        inst✝ : (i : ι) → Monoid (M i)
        x : Sigma fun i => M i
        l : List (Sigma fun i => M i)
        i j : ι
        w' : Monoid.CoprodI.NeWord M i j
        hnot1✝ : ∀ (l_1 : Sigma fun i => M i), Membership.mem (List.cons x (List.cons  …
        hnot1 : And (Ne x.snd 1) (∀ (x : Sigma fun i => M i), Membership.mem (List.con …
        hchain✝ : List.Chain' (fun l l' => Ne l.fst l'.fst) (List.cons x (List.cons ⟨i …
        hchain : And (Ne x.fst ⟨i, w'.head⟩.fst) (List.Chain' (fun l l' => Ne l.fst l' …
        h : Ne { toList := List.cons x (List.cons ⟨i, w'.head⟩ l), ne_one := hnot1✝, c …
        hw' : Eq w'.toList (List.cons ⟨i, w'.head⟩ l)
        ⊢ Exists fun i_1 => Exists fun j_1 => Exists fun w'_1 => Eq w'_1.toWord.toList …
      -/
      refine ⟨x.1, j, append (singleton x.2 hnot1.1) hchain.1 w', ?_⟩
      /-
        case mk.cons.cons.intro.intro.intro
        ι : Type u_1
        M : ι → Type u_2
        inst✝ : (i : ι) → Monoid (M i)
        x : Sigma fun i => M i
        l : List (Sigma fun i => M i)
        i j : ι
        w' : Monoid.CoprodI.NeWord M i j
        hnot1✝ : ∀ (l_1 : Sigma fun i => M i), Membership.mem (List.cons x (List.cons  …
        hnot1 : And (Ne x.snd 1) (∀ (x : Sigma fun i => M i), Membership.mem (List.con …
        hchain✝ : List.Chain' (fun l l' => Ne l.fst l'.fst) (List.cons x (List.cons ⟨i …
        hchain : And (Ne x.fst ⟨i, w'.head⟩.fst) (List.Chain' (fun l l' => Ne l.fst l' …
        h : Ne { toList := List.cons x (List.cons ⟨i, w'.head⟩ l), ne_one := hnot1✝, c …
        hw' : Eq w'.toList (List.cons ⟨i, w'.head⟩ l)
        ⊢ Eq ((Monoid.CoprodI.NeWord.singleton x.snd ⋯).append ⋯ w').toWord.toList { t …
      -/
      simpa [toWord] using hw'
      /-
        🎉 no goals
      -/


/-- A non-empty reduced word determines an element of the free product, given by multiplication. -/
def prod {i j} (w : NeWord M i j) :=
  w.toWord.prod


@[simp]
theorem singleton_head {i} (x : M i) (hne_one : x ≠ 1) : (singleton x hne_one).head = x :=
  rfl


@[simp]
theorem singleton_last {i} (x : M i) (hne_one : x ≠ 1) : (singleton x hne_one).last = x :=
  rfl


@[simp]
theorem prod_singleton {i} (x : M i) (hne_one : x ≠ 1) : (singleton x hne_one).prod = of x := by
  /-
    ι : Type u_1
    M : ι → Type u_2
    inst✝ : (i : ι) → Monoid (M i)
    i : ι
    x : M i
    hne_one : Ne x 1
    ⊢ Eq (Monoid.CoprodI.NeWord.singleton x hne_one).prod (Monoid.CoprodI.of x)
  -/
  simp [toWord, prod, Word.prod]
  /-
    🎉 no goals
  -/


@[simp]
theorem append_head {i j k l} {w₁ : NeWord M i j} {hne : j ≠ k} {w₂ : NeWord M k l} :
    (append w₁ hne w₂).head = w₁.head :=
  rfl


@[simp]
theorem append_last {i j k l} {w₁ : NeWord M i j} {hne : j ≠ k} {w₂ : NeWord M k l} :
    (append w₁ hne w₂).last = w₂.last :=
  rfl


@[simp]
theorem append_prod {i j k l} {w₁ : NeWord M i j} {hne : j ≠ k} {w₂ : NeWord M k l} :
                                                      /-
                                                        ι : Type u_1
                                                        M : ι → Type u_2
                                                        inst✝ : (i : ι) → Monoid (M i)
                                                        i j k l : ι
                                                        w₁ : Monoid.CoprodI.NeWord M i j
                                                        hne : Ne j k
                                                        w₂ : Monoid.CoprodI.NeWord M k l
                                                        ⊢ Eq (w₁.append hne w₂).prod (HMul.hMul w₁.prod w₂.prod)
                                                      -/
    (append w₁ hne w₂).prod = w₁.prod * w₂.prod := by simp [toWord, prod, Word.prod]
                                                      /-
                                                        🎉 no goals
                                                      -/


/-- One can replace the first letter in a non-empty reduced word by an element of the same
group -/
def replaceHead : ∀ {i j : ι} (x : M i) (_hnotone : x ≠ 1) (_w : NeWord M i j), NeWord M i j
  | _, _, x, h, singleton _ _ => singleton x h
  | _, _, x, h, append w₁ hne w₂ => append (replaceHead x h w₁) hne w₂


@[simp]
theorem replaceHead_head {i j : ι} (x : M i) (hnotone : x ≠ 1) (w : NeWord M i j) :
    (replaceHead x hnotone w).head = x := by
  /-
    ι : Type u_1
    M : ι → Type u_2
    inst✝ : (i : ι) → Monoid (M i)
    i j : ι
    x : M i
    hnotone : Ne x 1
    w : Monoid.CoprodI.NeWord M i j
    ⊢ Eq (Monoid.CoprodI.NeWord.replaceHead x hnotone w).head x
  -/
  induction w
    /-
      case singleton
      ι : Type u_1
      M : ι → Type u_2
      inst✝ : (i : ι) → Monoid (M i)
      i j i✝ : ι
      x✝ : M i✝
      a✝ : Ne x✝ 1
      x : M i✝
      hnotone : Ne x 1
      ⊢ Eq (Monoid.CoprodI.NeWord.replaceHead x hnotone (Monoid.CoprodI.NeWord.singl …
    -/
  · rfl
    /-
      🎉 no goals
    -/
    /-
      case append
      ι : Type u_1
      M : ι → Type u_2
      inst✝ : (i : ι) → Monoid (M i)
      i j i✝ j✝ k✝ l✝ : ι
      _w₁✝ : Monoid.CoprodI.NeWord M i✝ j✝
      _hne✝ : Ne j✝ k✝
      _w₂✝ : Monoid.CoprodI.NeWord M k✝ l✝
      _w₁_ih✝ : ∀ (x : M i✝) (hnotone : Ne x 1), Eq (Monoid.CoprodI.NeWord.replaceHe …
      _w₂_ih✝ : ∀ (x : M k✝) (hnotone : Ne x 1), Eq (Monoid.CoprodI.NeWord.replaceHe …
      x : M i✝
      hnotone : Ne x 1
      ⊢ Eq (Monoid.CoprodI.NeWord.replaceHead x hnotone (_w₁✝.append _hne✝ _w₂✝)).he …
    -/
  · simp [*, replaceHead]
    /-
      🎉 no goals
    -/


/-- One can multiply an element from the left to a non-empty reduced word if it does not cancel
with the first element in the word. -/
def mulHead {i j : ι} (w : NeWord M i j) (x : M i) (hnotone : x * w.head ≠ 1) : NeWord M i j :=
  replaceHead (x * w.head) hnotone w


@[simp]
theorem mulHead_head {i j : ι} (w : NeWord M i j) (x : M i) (hnotone : x * w.head ≠ 1) :
    (mulHead w x hnotone).head = x * w.head := by
  /-
    ι : Type u_1
    M : ι → Type u_2
    inst✝ : (i : ι) → Monoid (M i)
    i j : ι
    w : Monoid.CoprodI.NeWord M i j
    x : M i
    hnotone : Ne (HMul.hMul x w.head) 1
    ⊢ Eq (w.mulHead x hnotone).head (HMul.hMul x w.head)
  -/
  induction w
    /-
      case singleton
      ι : Type u_1
      M : ι → Type u_2
      inst✝ : (i : ι) → Monoid (M i)
      i j i✝ : ι
      x✝ : M i✝
      a✝ : Ne x✝ 1
      x : M i✝
      hnotone : Ne (HMul.hMul x (Monoid.CoprodI.NeWord.singleton x✝ a✝).head) 1
      ⊢ Eq ((Monoid.CoprodI.NeWord.singleton x✝ a✝).mulHead x hnotone).head (HMul.hM …
    -/
  · rfl
    /-
      🎉 no goals
    -/
    /-
      case append
      ι : Type u_1
      M : ι → Type u_2
      inst✝ : (i : ι) → Monoid (M i)
      i j i✝ j✝ k✝ l✝ : ι
      _w₁✝ : Monoid.CoprodI.NeWord M i✝ j✝
      _hne✝ : Ne j✝ k✝
      _w₂✝ : Monoid.CoprodI.NeWord M k✝ l✝
      _w₁_ih✝ : ∀ (x : M i✝) (hnotone : Ne (HMul.hMul x _w₁✝.head) 1), Eq (_w₁✝.mulH …
      _w₂_ih✝ : ∀ (x : M k✝) (hnotone : Ne (HMul.hMul x _w₂✝.head) 1), Eq (_w₂✝.mulH …
      x : M i✝
      hnotone : Ne (HMul.hMul x (_w₁✝.append _hne✝ _w₂✝).head) 1
      ⊢ Eq ((_w₁✝.append _hne✝ _w₂✝).mulHead x hnotone).head (HMul.hMul x (_w₁✝.appe …
    -/
  · simp [*, mulHead]
    /-
      🎉 no goals
    -/


@[simp]
theorem mulHead_prod {i j : ι} (w : NeWord M i j) (x : M i) (hnotone : x * w.head ≠ 1) :
    (mulHead w x hnotone).prod = of x * w.prod := by
  /-
    ι : Type u_1
    M : ι → Type u_2
    inst✝ : (i : ι) → Monoid (M i)
    i j : ι
    w : Monoid.CoprodI.NeWord M i j
    x : M i
    hnotone : Ne (HMul.hMul x w.head) 1
    ⊢ Eq (w.mulHead x hnotone).prod (HMul.hMul (Monoid.CoprodI.of x) w.prod)
  -/
  unfold mulHead
  induction w with
  | singleton => simp [mulHead, replaceHead]
  | append _ _ _ w_ih_w₁ w_ih_w₂ =>
    specialize w_ih_w₁ _ hnotone
    clear w_ih_w₂
    simp? [replaceHead, ← mul_assoc] at * says
      simp only [replaceHead, head, append_prod, ← mul_assoc] at *
    congr 1


/-- The inverse of a non-empty reduced word -/
def inv : ∀ {i j} (_w : NeWord G i j), NeWord G j i
  | _, _, singleton x h => singleton x⁻¹ (mt inv_eq_one.mp h)
  | _, _, append w₁ h w₂ => append w₂.inv h.symm w₁.inv


@[simp]
theorem inv_prod {i j} (w : NeWord G i j) : w.inv.prod = w.prod⁻¹ := by
  /-
    ι : Type u_1
    G : ι → Type u_4
    inst✝ : (i : ι) → Group (G i)
    i j : ι
    w : Monoid.CoprodI.NeWord G i j
    ⊢ Eq w.inv.prod (Inv.inv w.prod)
  -/
                  /-
                    🎉 no goals
                  -/
  induction w <;> simp [inv, *]
                  /-
                    🎉 no goals
                  -/


@[simp]
theorem inv_head {i j} (w : NeWord G i j) : w.inv.head = w.last⁻¹ := by
  /-
    ι : Type u_1
    G : ι → Type u_4
    inst✝ : (i : ι) → Group (G i)
    i j : ι
    w : Monoid.CoprodI.NeWord G i j
    ⊢ Eq w.inv.head (Inv.inv w.last)
  -/
                  /-
                    🎉 no goals
                  -/
  induction w <;> simp [inv, *]
                  /-
                    🎉 no goals
                  -/


@[simp]
theorem inv_last {i j} (w : NeWord G i j) : w.inv.last = w.head⁻¹ := by
  /-
    ι : Type u_1
    G : ι → Type u_4
    inst✝ : (i : ι) → Group (G i)
    i j : ι
    w : Monoid.CoprodI.NeWord G i j
    ⊢ Eq w.inv.last (Inv.inv w.head)
  -/
                  /-
                    🎉 no goals
                  -/
  induction w <;> simp [inv, *]
                  /-
                    🎉 no goals
                  -/


theorem lift_word_ping_pong {i j k} (w : NeWord H i j) (hk : j ≠ k) :
    lift f w.prod • X k ⊆ X i := by
  /-
    ι : Type u_1
    G : Type u_4
    inst✝² : Group G
    H : ι → Type u_5
    inst✝¹ : (i : ι) → Group (H i)
    f : (i : ι) → MonoidHom (H i) G
    α : Type u_6
    inst✝ : MulAction G α
    X : ι → Set α
    hpp : Pairwise fun i j => ∀ (h : H i), Ne h 1 → HasSubset.Subset (HSMul.hSMul  …
    i j k : ι
    w : Monoid.CoprodI.NeWord H i j
    hk : Ne j k
    ⊢ HasSubset.Subset (HSMul.hSMul ((Monoid.CoprodI.lift f) w.prod) (X k)) (X i)
  -/
  induction' w with i x hne_one i j k l w₁ hne w₂ hIw₁ hIw₂ generalizing k
    /-
      case singleton
      ι : Type u_1
      G : Type u_4
      inst✝² : Group G
      H : ι → Type u_5
      inst✝¹ : (i : ι) → Group (H i)
      f : (i : ι) → MonoidHom (H i) G
      α : Type u_6
      inst✝ : MulAction G α
      X : ι → Set α
      hpp : Pairwise fun i j => ∀ (h : H i), Ne h 1 → HasSubset.Subset (HSMul.hSMul  …
      i✝ j i : ι
      x : H i
      hne_one : Ne x 1
      k : ι
      hk : Ne i k
      ⊢ HasSubset.Subset (HSMul.hSMul ((Monoid.CoprodI.lift f) (Monoid.CoprodI.NeWor …
    -/
  · simpa using hpp hk _ hne_one
    /-
      🎉 no goals
    -/
  · calc
      lift f (NeWord.append w₁ hne w₂).prod • X k = lift f w₁.prod • lift f w₂.prod • X k := by
        simp [MulAction.mul_smul]
      _ ⊆ lift f w₁.prod • X _ := set_smul_subset_set_smul_iff.mpr (hIw₂ hk)
      _ ⊆ X i := hIw₁ hne


theorem lift_word_prod_nontrivial_of_other_i {i j k} (w : NeWord H i j) (hhead : k ≠ i)
    (hlast : k ≠ j) : lift f w.prod ≠ 1 := by
  /-
    ι : Type u_1
    G : Type u_4
    inst✝² : Group G
    H : ι → Type u_5
    inst✝¹ : (i : ι) → Group (H i)
    f : (i : ι) → MonoidHom (H i) G
    α : Type u_6
    inst✝ : MulAction G α
    X : ι → Set α
    hXnonempty : ∀ (i : ι), (X i).Nonempty
    hXdisj : Pairwise (Function.onFun Disjoint X)
    hpp : Pairwise fun i j => ∀ (h : H i), Ne h 1 → HasSubset.Subset (HSMul.hSMul  …
    i j k : ι
    w : Monoid.CoprodI.NeWord H i j
    hhead : Ne k i
    hlast : Ne k j
    ⊢ Ne ((Monoid.CoprodI.lift f) w.prod) 1
  -/
  intro heq1
  /-
    ι : Type u_1
    G : Type u_4
    inst✝² : Group G
    H : ι → Type u_5
    inst✝¹ : (i : ι) → Group (H i)
    f : (i : ι) → MonoidHom (H i) G
    α : Type u_6
    inst✝ : MulAction G α
    X : ι → Set α
    hXnonempty : ∀ (i : ι), (X i).Nonempty
    hXdisj : Pairwise (Function.onFun Disjoint X)
    hpp : Pairwise fun i j => ∀ (h : H i), Ne h 1 → HasSubset.Subset (HSMul.hSMul  …
    i j k : ι
    w : Monoid.CoprodI.NeWord H i j
    hhead : Ne k i
    hlast : Ne k j
    heq1 : Eq ((Monoid.CoprodI.lift f) w.prod) 1
    ⊢ False
  -/
  have : X k ⊆ X i := by simpa [heq1] using lift_word_ping_pong f X hpp w hlast.symm
  /-
    ι : Type u_1
    G : Type u_4
    inst✝² : Group G
    H : ι → Type u_5
    inst✝¹ : (i : ι) → Group (H i)
    f : (i : ι) → MonoidHom (H i) G
    α : Type u_6
    inst✝ : MulAction G α
    X : ι → Set α
    hXnonempty : ∀ (i : ι), (X i).Nonempty
    hXdisj : Pairwise (Function.onFun Disjoint X)
    hpp : Pairwise fun i j => ∀ (h : H i), Ne h 1 → HasSubset.Subset (HSMul.hSMul  …
    i j k : ι
    w : Monoid.CoprodI.NeWord H i j
    hhead : Ne k i
    hlast : Ne k j
    heq1 : Eq ((Monoid.CoprodI.lift f) w.prod) 1
    this : HasSubset.Subset (X k) (X i)
    ⊢ False
  -/
  obtain ⟨x, hx⟩ := hXnonempty k
  /-
    case intro
    ι : Type u_1
    G : Type u_4
    inst✝² : Group G
    H : ι → Type u_5
    inst✝¹ : (i : ι) → Group (H i)
    f : (i : ι) → MonoidHom (H i) G
    α : Type u_6
    inst✝ : MulAction G α
    X : ι → Set α
    hXnonempty : ∀ (i : ι), (X i).Nonempty
    hXdisj : Pairwise (Function.onFun Disjoint X)
    hpp : Pairwise fun i j => ∀ (h : H i), Ne h 1 → HasSubset.Subset (HSMul.hSMul  …
    i j k : ι
    w : Monoid.CoprodI.NeWord H i j
    hhead : Ne k i
    hlast : Ne k j
    heq1 : Eq ((Monoid.CoprodI.lift f) w.prod) 1
    this : HasSubset.Subset (X k) (X i)
    x : α
    hx : Membership.mem (X k) x
    ⊢ False
  -/
  exact (hXdisj hhead).le_bot ⟨hx, this hx⟩
  /-
    🎉 no goals
  -/


theorem lift_word_prod_nontrivial_of_head_eq_last {i} (w : NeWord H i i) :
    lift f w.prod ≠ 1 := by
  /-
    ι : Type u_1
    G : Type u_4
    inst✝³ : Group G
    H : ι → Type u_5
    inst✝² : (i : ι) → Group (H i)
    f : (i : ι) → MonoidHom (H i) G
    α : Type u_6
    inst✝¹ : MulAction G α
    X : ι → Set α
    hXnonempty : ∀ (i : ι), (X i).Nonempty
    hXdisj : Pairwise (Function.onFun Disjoint X)
    hpp : Pairwise fun i j => ∀ (h : H i), Ne h 1 → HasSubset.Subset (HSMul.hSMul  …
    inst✝ : Nontrivial ι
    i : ι
    w : Monoid.CoprodI.NeWord H i i
    ⊢ Ne ((Monoid.CoprodI.lift f) w.prod) 1
  -/
  obtain ⟨k, hk⟩ := exists_ne i
  /-
    case intro
    ι : Type u_1
    G : Type u_4
    inst✝³ : Group G
    H : ι → Type u_5
    inst✝² : (i : ι) → Group (H i)
    f : (i : ι) → MonoidHom (H i) G
    α : Type u_6
    inst✝¹ : MulAction G α
    X : ι → Set α
    hXnonempty : ∀ (i : ι), (X i).Nonempty
    hXdisj : Pairwise (Function.onFun Disjoint X)
    hpp : Pairwise fun i j => ∀ (h : H i), Ne h 1 → HasSubset.Subset (HSMul.hSMul  …
    inst✝ : Nontrivial ι
    i : ι
    w : Monoid.CoprodI.NeWord H i i
    k : ι
    hk : Ne k i
    ⊢ Ne ((Monoid.CoprodI.lift f) w.prod) 1
  -/
  exact lift_word_prod_nontrivial_of_other_i f X hXnonempty hXdisj hpp w hk hk
  /-
    🎉 no goals
  -/


theorem lift_word_prod_nontrivial_of_head_card {i j} (w : NeWord H i j)
    (hcard : 3 ≤ #(H i)) (hheadtail : i ≠ j) : lift f w.prod ≠ 1 := by
  /-
    ι : Type u_1
    G : Type u_4
    inst✝³ : Group G
    H : ι → Type u_5
    inst✝² : (i : ι) → Group (H i)
    f : (i : ι) → MonoidHom (H i) G
    α : Type u_6
    inst✝¹ : MulAction G α
    X : ι → Set α
    hXnonempty : ∀ (i : ι), (X i).Nonempty
    hXdisj : Pairwise (Function.onFun Disjoint X)
    hpp : Pairwise fun i j => ∀ (h : H i), Ne h 1 → HasSubset.Subset (HSMul.hSMul  …
    inst✝ : Nontrivial ι
    i j : ι
    w : Monoid.CoprodI.NeWord H i j
    hcard : LE.le 3 (Cardinal.mk (H i))
    hheadtail : Ne i j
    ⊢ Ne ((Monoid.CoprodI.lift f) w.prod) 1
  -/
  obtain ⟨h, hn1, hnh⟩ := Cardinal.three_le hcard 1 w.head⁻¹
  have hnot1 : h * w.head ≠ 1 := by
    rw [← div_inv_eq_mul]
    exact div_ne_one_of_ne hnh
  let w' : NeWord H i i :=
    NeWord.append (NeWord.mulHead w h hnot1) hheadtail.symm
      (NeWord.singleton h⁻¹ (inv_ne_one.mpr hn1))
  have hw' : lift f w'.prod ≠ 1 :=
    lift_word_prod_nontrivial_of_head_eq_last f X hXnonempty hXdisj hpp w'
  /-
    case intro.intro
    ι : Type u_1
    G : Type u_4
    inst✝³ : Group G
    H : ι → Type u_5
    inst✝² : (i : ι) → Group (H i)
    f : (i : ι) → MonoidHom (H i) G
    α : Type u_6
    inst✝¹ : MulAction G α
    X : ι → Set α
    hXnonempty : ∀ (i : ι), (X i).Nonempty
    hXdisj : Pairwise (Function.onFun Disjoint X)
    hpp : Pairwise fun i j => ∀ (h : H i), Ne h 1 → HasSubset.Subset (HSMul.hSMul  …
    inst✝ : Nontrivial ι
    i j : ι
    w : Monoid.CoprodI.NeWord H i j
    hcard : LE.le 3 (Cardinal.mk (H i))
    hheadtail : Ne i j
    h : H i
    hn1 : Ne h 1
    hnh : Ne h (Inv.inv w.head)
    hnot1 : Ne (HMul.hMul h w.head) 1
    w' : Monoid.CoprodI.NeWord H i i := (w.mulHead h hnot1).append ⋯ (Monoid.Copro …
    hw' : Ne ((Monoid.CoprodI.lift f) w'.prod) 1
    ⊢ Ne ((Monoid.CoprodI.lift f) w.prod) 1
  -/
  intro heq1
  /-
    case intro.intro
    ι : Type u_1
    G : Type u_4
    inst✝³ : Group G
    H : ι → Type u_5
    inst✝² : (i : ι) → Group (H i)
    f : (i : ι) → MonoidHom (H i) G
    α : Type u_6
    inst✝¹ : MulAction G α
    X : ι → Set α
    hXnonempty : ∀ (i : ι), (X i).Nonempty
    hXdisj : Pairwise (Function.onFun Disjoint X)
    hpp : Pairwise fun i j => ∀ (h : H i), Ne h 1 → HasSubset.Subset (HSMul.hSMul  …
    inst✝ : Nontrivial ι
    i j : ι
    w : Monoid.CoprodI.NeWord H i j
    hcard : LE.le 3 (Cardinal.mk (H i))
    hheadtail : Ne i j
    h : H i
    hn1 : Ne h 1
    hnh : Ne h (Inv.inv w.head)
    hnot1 : Ne (HMul.hMul h w.head) 1
    w' : Monoid.CoprodI.NeWord H i i := (w.mulHead h hnot1).append ⋯ (Monoid.Copro …
    hw' : Ne ((Monoid.CoprodI.lift f) w'.prod) 1
    heq1 : Eq ((Monoid.CoprodI.lift f) w.prod) 1
    ⊢ False
  -/
  apply hw'
  /-
    case intro.intro
    ι : Type u_1
    G : Type u_4
    inst✝³ : Group G
    H : ι → Type u_5
    inst✝² : (i : ι) → Group (H i)
    f : (i : ι) → MonoidHom (H i) G
    α : Type u_6
    inst✝¹ : MulAction G α
    X : ι → Set α
    hXnonempty : ∀ (i : ι), (X i).Nonempty
    hXdisj : Pairwise (Function.onFun Disjoint X)
    hpp : Pairwise fun i j => ∀ (h : H i), Ne h 1 → HasSubset.Subset (HSMul.hSMul  …
    inst✝ : Nontrivial ι
    i j : ι
    w : Monoid.CoprodI.NeWord H i j
    hcard : LE.le 3 (Cardinal.mk (H i))
    hheadtail : Ne i j
    h : H i
    hn1 : Ne h 1
    hnh : Ne h (Inv.inv w.head)
    hnot1 : Ne (HMul.hMul h w.head) 1
    w' : Monoid.CoprodI.NeWord H i i := (w.mulHead h hnot1).append ⋯ (Monoid.Copro …
    hw' : Ne ((Monoid.CoprodI.lift f) w'.prod) 1
    heq1 : Eq ((Monoid.CoprodI.lift f) w.prod) 1
    ⊢ Eq ((Monoid.CoprodI.lift f) w'.prod) 1
  -/
  simp [w', heq1]
  /-
    🎉 no goals
  -/


include hcard in
theorem lift_word_prod_nontrivial_of_not_empty {i j} (w : NeWord H i j) :
    lift f w.prod ≠ 1 := by
  classical
    cases' hcard with hcard hcard
    · obtain ⟨i, h1, h2⟩ := Cardinal.three_le hcard i j
      exact lift_word_prod_nontrivial_of_other_i f X hXnonempty hXdisj hpp w h1 h2
    · cases' hcard with k hcard
      by_cases hh : i = k <;> by_cases hl : j = k
      · subst hh
        subst hl
        exact lift_word_prod_nontrivial_of_head_eq_last f X hXnonempty hXdisj hpp w
      · subst hh
        change j ≠ i at hl
        exact lift_word_prod_nontrivial_of_head_card f X hXnonempty hXdisj hpp w hcard hl.symm
      · subst hl
        change i ≠ j at hh
        have : lift f w.inv.prod ≠ 1 :=
          lift_word_prod_nontrivial_of_head_card f X hXnonempty hXdisj hpp w.inv hcard hh.symm
        intro heq
        apply this
        simpa using heq
      · change i ≠ k at hh
        change j ≠ k at hl
        obtain ⟨h, hn1, -⟩ := Cardinal.three_le hcard 1 1
        let w' : NeWord H k k :=
          NeWord.append (NeWord.append (NeWord.singleton h hn1) hh.symm w) hl
            (NeWord.singleton h⁻¹ (inv_ne_one.mpr hn1))
        have hw' : lift f w'.prod ≠ 1 :=
          lift_word_prod_nontrivial_of_head_eq_last f X hXnonempty hXdisj hpp w'
        intro heq1
        apply hw'
        simp [w', heq1]


include hcard in
theorem empty_of_word_prod_eq_one {w : Word H} (h : lift f w.prod = 1) :
    w = Word.empty := by
  /-
    ι : Type u_1
    G : Type u_4
    inst✝³ : Group G
    H : ι → Type u_5
    inst✝² : (i : ι) → Group (H i)
    f : (i : ι) → MonoidHom (H i) G
    hcard : Or (LE.le 3 (Cardinal.mk ι)) (Exists fun i => LE.le 3 (Cardinal.mk (H  …
    α : Type u_6
    inst✝¹ : MulAction G α
    X : ι → Set α
    hXnonempty : ∀ (i : ι), (X i).Nonempty
    hXdisj : Pairwise (Function.onFun Disjoint X)
    hpp : Pairwise fun i j => ∀ (h : H i), Ne h 1 → HasSubset.Subset (HSMul.hSMul  …
    inst✝ : Nontrivial ι
    w : Monoid.CoprodI.Word H
    h : Eq ((Monoid.CoprodI.lift f) w.prod) 1
    ⊢ Eq w Monoid.CoprodI.Word.empty
  -/
  by_contra hnotempty
  /-
    ι : Type u_1
    G : Type u_4
    inst✝³ : Group G
    H : ι → Type u_5
    inst✝² : (i : ι) → Group (H i)
    f : (i : ι) → MonoidHom (H i) G
    hcard : Or (LE.le 3 (Cardinal.mk ι)) (Exists fun i => LE.le 3 (Cardinal.mk (H  …
    α : Type u_6
    inst✝¹ : MulAction G α
    X : ι → Set α
    hXnonempty : ∀ (i : ι), (X i).Nonempty
    hXdisj : Pairwise (Function.onFun Disjoint X)
    hpp : Pairwise fun i j => ∀ (h : H i), Ne h 1 → HasSubset.Subset (HSMul.hSMul  …
    inst✝ : Nontrivial ι
    w : Monoid.CoprodI.Word H
    h : Eq ((Monoid.CoprodI.lift f) w.prod) 1
    hnotempty : Not (Eq w Monoid.CoprodI.Word.empty)
    ⊢ False
  -/
  obtain ⟨i, j, w, rfl⟩ := NeWord.of_word w hnotempty
  /-
    case intro.intro.intro
    ι : Type u_1
    G : Type u_4
    inst✝³ : Group G
    H : ι → Type u_5
    inst✝² : (i : ι) → Group (H i)
    f : (i : ι) → MonoidHom (H i) G
    hcard : Or (LE.le 3 (Cardinal.mk ι)) (Exists fun i => LE.le 3 (Cardinal.mk (H  …
    α : Type u_6
    inst✝¹ : MulAction G α
    X : ι → Set α
    hXnonempty : ∀ (i : ι), (X i).Nonempty
    hXdisj : Pairwise (Function.onFun Disjoint X)
    hpp : Pairwise fun i j => ∀ (h : H i), Ne h 1 → HasSubset.Subset (HSMul.hSMul  …
    inst✝ : Nontrivial ι
    i j : ι
    w : Monoid.CoprodI.NeWord H i j
    h : Eq ((Monoid.CoprodI.lift f) w.toWord.prod) 1
    hnotempty : Not (Eq w.toWord Monoid.CoprodI.Word.empty)
    ⊢ False
  -/
  exact lift_word_prod_nontrivial_of_not_empty f hcard X hXnonempty hXdisj hpp w h
  /-
    🎉 no goals
  -/


include hcard in
/-- The **Ping-Pong-Lemma**.

Given a group action of `G` on `X` so that the `H i` acts in a specific way on disjoint subsets
`X i` we can prove that `lift f` is injective, and thus the image of `lift f` is isomorphic to the
free product of the `H i`.

Often the Ping-Pong-Lemma is stated with regard to subgroups `H i` that generate the whole group;
we generalize to arbitrary group homomorphisms `f i : H i →* G` and do not require the group to be
generated by the images.

Usually the Ping-Pong-Lemma requires that one group `H i` has at least three elements. This
condition is only needed if `# ι = 2`, and we accept `3 ≤ # ι` as an alternative.
-/
theorem lift_injective_of_ping_pong : Function.Injective (lift f) := by
  classical
    apply (injective_iff_map_eq_one (lift f)).mpr
    rw [(CoprodI.Word.equiv).forall_congr_left]
    intro w Heq
    dsimp [Word.equiv] at *
    rw [empty_of_word_prod_eq_one f hcard X hXnonempty hXdisj hpp Heq, Word.prod_empty]


/-- Given a family of free groups with distinguished bases, then their free product is free, with
a basis given by the union of the bases of the components. -/
def FreeGroupBasis.coprodI {ι : Type*} {X : ι → Type*} {G : ι → Type*} [∀ i, Group (G i)]
    (B : ∀ i, FreeGroupBasis (X i) (G i)) :
    FreeGroupBasis (Σ i, X i) (CoprodI G) :=
  ⟨MulEquiv.symm <| MonoidHom.toMulEquiv
    (FreeGroup.lift fun x : Σ i, X i => CoprodI.of (B x.1 x.2))
    (CoprodI.lift fun i : ι => (B i).lift fun x : X i =>
              FreeGroup.of (⟨i, x⟩ : Σ i, X i))
        /-
          ι✝ : Type u_1
          M : ι✝ → Type u_2
          inst✝² : (i : ι✝) → Monoid (M i)
          N : Type u_3
          inst✝¹ : Monoid N
          ι : Type u_4
          X : ι → Type u_5
          G : ι → Type u_6
          inst✝ : (i : ι) → Group (G i)
          B : (i : ι) → FreeGroupBasis (X i) (G i)
          ⊢ Eq ((Monoid.CoprodI.lift fun i => (B i).lift fun x => FreeGroup.of ⟨i, x⟩).c …
        -/
    (by ext; simp)
             /-
               🎉 no goals
             -/
        /-
          ι✝ : Type u_1
          M : ι✝ → Type u_2
          inst✝² : (i : ι✝) → Monoid (M i)
          N : Type u_3
          inst✝¹ : Monoid N
          ι : Type u_4
          X : ι → Type u_5
          G : ι → Type u_6
          inst✝ : (i : ι) → Group (G i)
          B : (i : ι) → FreeGroupBasis (X i) (G i)
          ⊢ Eq ((FreeGroup.lift fun x => Monoid.CoprodI.of ((B x.fst) x.snd)).comp (Mono …
        -/
    (by ext1 i; apply (B i).ext_hom; simp)⟩
                                     /-
                                       🎉 no goals
                                     -/


/-- The free product of free groups is itself a free group. -/
instance {ι : Type*} (G : ι → Type*) [∀ i, Group (G i)] [∀ i, IsFreeGroup (G i)] :
    IsFreeGroup (CoprodI G) :=
  (FreeGroupBasis.coprodI (fun i ↦ IsFreeGroup.basis (G i))).isFreeGroup

-- NB: One might expect this theorem to be phrased with ℤ, but ℤ is an additive group,
-- and using `Multiplicative ℤ` runs into diamond issues.

/-- A free group is a free product of copies of the free_group over one generator. -/
@[simps!]
def _root_.freeGroupEquivCoprodI {ι : Type u_1} :
    FreeGroup ι ≃* CoprodI fun _ : ι => FreeGroup Unit := by
  /-
    ι✝ : Type u_1
    M : ι✝ → Type u_2
    inst✝¹ : (i : ι✝) → Monoid (M i)
    N : Type u_3
    inst✝ : Monoid N
    ι : Type u_1
    ⊢ MulEquiv (FreeGroup ι) (Monoid.CoprodI fun x => FreeGroup Unit)
  -/
  refine MonoidHom.toMulEquiv ?_ ?_ ?_ ?_
    /-
      case refine_1
      ι✝ : Type u_1
      M : ι✝ → Type u_2
      inst✝¹ : (i : ι✝) → Monoid (M i)
      N : Type u_3
      inst✝ : Monoid N
      ι : Type u_1
      ⊢ MonoidHom (FreeGroup ι) (Monoid.CoprodI fun x => FreeGroup Unit)
    -/
  · exact FreeGroup.lift fun i => @CoprodI.of ι _ _ i (FreeGroup.of Unit.unit)
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      ι✝ : Type u_1
      M : ι✝ → Type u_2
      inst✝¹ : (i : ι✝) → Monoid (M i)
      N : Type u_3
      inst✝ : Monoid N
      ι : Type u_1
      ⊢ MonoidHom (Monoid.CoprodI fun x => FreeGroup Unit) (FreeGroup ι)
    -/
  · exact CoprodI.lift fun i => FreeGroup.lift fun _ => FreeGroup.of i
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      ι✝ : Type u_1
      M : ι✝ → Type u_2
      inst✝¹ : (i : ι✝) → Monoid (M i)
      N : Type u_3
      inst✝ : Monoid N
      ι : Type u_1
      ⊢ Eq ((Monoid.CoprodI.lift fun i => FreeGroup.lift fun x => FreeGroup.of i).co …
    -/
  · ext; simp
         /-
           🎉 no goals
         -/
    /-
      case refine_4
      ι✝ : Type u_1
      M : ι✝ → Type u_2
      inst✝¹ : (i : ι✝) → Monoid (M i)
      N : Type u_3
      inst✝ : Monoid N
      ι : Type u_1
      ⊢ Eq ((FreeGroup.lift fun i => Monoid.CoprodI.of (FreeGroup.of Unit.unit)).com …
    -/
  · ext i a; cases a; simp
                      /-
                        🎉 no goals
                      -/


include hXnonempty hXdisj hYdisj hXYdisj hX hY in
/-- The Ping-Pong-Lemma.

Given a group action of `G` on `X` so that the generators of the free groups act in specific
ways on disjoint subsets `X i` and `Y i` we can prove that `lift f` is injective, and thus the image
of `lift f` is isomorphic to the free group.

Often the Ping-Pong-Lemma is stated with regard to group elements that generate the whole group;
we generalize to arbitrary group homomorphisms from the free group to `G` and do not require the
group to be generated by the elements.
-/
theorem _root_.FreeGroup.injective_lift_of_ping_pong : Function.Injective (FreeGroup.lift a) := by
  -- Step one: express the free group lift via the free product lift
  have : FreeGroup.lift a =
      (CoprodI.lift fun i => FreeGroup.lift fun _ => a i).comp
        (@freeGroupEquivCoprodI ι).toMonoidHom := by
    ext i
    simp
  /-
    ι : Type u_1
    inst✝² : Nontrivial ι
    G : Type u_1
    inst✝¹ : Group G
    a : ι → G
    α : Type u_4
    inst✝ : MulAction G α
    X Y : ι → Set α
    hXnonempty : ∀ (i : ι), (X i).Nonempty
    hXdisj : Pairwise (Function.onFun Disjoint X)
    hYdisj : Pairwise (Function.onFun Disjoint Y)
    hXYdisj : ∀ (i j : ι), Disjoint (X i) (Y j)
    hX : ∀ (i : ι), HasSubset.Subset (HSMul.hSMul (a i) (HasCompl.compl (Y i))) (X …
    hY : ∀ (i : ι), HasSubset.Subset (HSMul.hSMul (Inv.inv a i) (HasCompl.compl (X …
    this : Eq (FreeGroup.lift a) ((Monoid.CoprodI.lift fun i => FreeGroup.lift fun …
    ⊢ Function.Injective ⇑(FreeGroup.lift a)
  -/
  rw [this, MonoidHom.coe_comp]
  /-
    ι : Type u_1
    inst✝² : Nontrivial ι
    G : Type u_1
    inst✝¹ : Group G
    a : ι → G
    α : Type u_4
    inst✝ : MulAction G α
    X Y : ι → Set α
    hXnonempty : ∀ (i : ι), (X i).Nonempty
    hXdisj : Pairwise (Function.onFun Disjoint X)
    hYdisj : Pairwise (Function.onFun Disjoint Y)
    hXYdisj : ∀ (i j : ι), Disjoint (X i) (Y j)
    hX : ∀ (i : ι), HasSubset.Subset (HSMul.hSMul (a i) (HasCompl.compl (Y i))) (X …
    hY : ∀ (i : ι), HasSubset.Subset (HSMul.hSMul (Inv.inv a i) (HasCompl.compl (X …
    this : Eq (FreeGroup.lift a) ((Monoid.CoprodI.lift fun i => FreeGroup.lift fun …
    ⊢ Function.Injective (Function.comp ⇑(Monoid.CoprodI.lift fun i => FreeGroup.l …
  -/
  clear this
  /-
    ι : Type u_1
    inst✝² : Nontrivial ι
    G : Type u_1
    inst✝¹ : Group G
    a : ι → G
    α : Type u_4
    inst✝ : MulAction G α
    X Y : ι → Set α
    hXnonempty : ∀ (i : ι), (X i).Nonempty
    hXdisj : Pairwise (Function.onFun Disjoint X)
    hYdisj : Pairwise (Function.onFun Disjoint Y)
    hXYdisj : ∀ (i j : ι), Disjoint (X i) (Y j)
    hX : ∀ (i : ι), HasSubset.Subset (HSMul.hSMul (a i) (HasCompl.compl (Y i))) (X …
    hY : ∀ (i : ι), HasSubset.Subset (HSMul.hSMul (Inv.inv a i) (HasCompl.compl (X …
    ⊢ Function.Injective (Function.comp ⇑(Monoid.CoprodI.lift fun i => FreeGroup.l …
  -/
  refine Function.Injective.comp ?_ (MulEquiv.injective freeGroupEquivCoprodI)
  -- Step two: Invoke the ping-pong lemma for free products
  /-
    ι : Type u_1
    inst✝² : Nontrivial ι
    G : Type u_1
    inst✝¹ : Group G
    a : ι → G
    α : Type u_4
    inst✝ : MulAction G α
    X Y : ι → Set α
    hXnonempty : ∀ (i : ι), (X i).Nonempty
    hXdisj : Pairwise (Function.onFun Disjoint X)
    hYdisj : Pairwise (Function.onFun Disjoint Y)
    hXYdisj : ∀ (i j : ι), Disjoint (X i) (Y j)
    hX : ∀ (i : ι), HasSubset.Subset (HSMul.hSMul (a i) (HasCompl.compl (Y i))) (X …
    hY : ∀ (i : ι), HasSubset.Subset (HSMul.hSMul (Inv.inv a i) (HasCompl.compl (X …
    ⊢ Function.Injective ⇑(Monoid.CoprodI.lift fun i => FreeGroup.lift fun x => a i)
  -/
  show Function.Injective (lift fun i : ι => FreeGroup.lift fun _ => a i)
  -- Prepare to instantiate lift_injective_of_ping_pong
  /-
    ι : Type u_1
    inst✝² : Nontrivial ι
    G : Type u_1
    inst✝¹ : Group G
    a : ι → G
    α : Type u_4
    inst✝ : MulAction G α
    X Y : ι → Set α
    hXnonempty : ∀ (i : ι), (X i).Nonempty
    hXdisj : Pairwise (Function.onFun Disjoint X)
    hYdisj : Pairwise (Function.onFun Disjoint Y)
    hXYdisj : ∀ (i j : ι), Disjoint (X i) (Y j)
    hX : ∀ (i : ι), HasSubset.Subset (HSMul.hSMul (a i) (HasCompl.compl (Y i))) (X …
    hY : ∀ (i : ι), HasSubset.Subset (HSMul.hSMul (Inv.inv a i) (HasCompl.compl (X …
    ⊢ Function.Injective ⇑(Monoid.CoprodI.lift fun i => FreeGroup.lift fun x => a i)
  -/
  let H : ι → Type _ := fun _i => FreeGroup Unit
  /-
    ι : Type u_1
    inst✝² : Nontrivial ι
    G : Type u_1
    inst✝¹ : Group G
    a : ι → G
    α : Type u_4
    inst✝ : MulAction G α
    X Y : ι → Set α
    hXnonempty : ∀ (i : ι), (X i).Nonempty
    hXdisj : Pairwise (Function.onFun Disjoint X)
    hYdisj : Pairwise (Function.onFun Disjoint Y)
    hXYdisj : ∀ (i j : ι), Disjoint (X i) (Y j)
    hX : ∀ (i : ι), HasSubset.Subset (HSMul.hSMul (a i) (HasCompl.compl (Y i))) (X …
    hY : ∀ (i : ι), HasSubset.Subset (HSMul.hSMul (Inv.inv a i) (HasCompl.compl (X …
    H : ι → Type := fun _i => FreeGroup Unit
    ⊢ Function.Injective ⇑(Monoid.CoprodI.lift fun i => FreeGroup.lift fun x => a i)
  -/
  let f : ∀ i, H i →* G := fun i => FreeGroup.lift fun _ => a i
  /-
    ι : Type u_1
    inst✝² : Nontrivial ι
    G : Type u_1
    inst✝¹ : Group G
    a : ι → G
    α : Type u_4
    inst✝ : MulAction G α
    X Y : ι → Set α
    hXnonempty : ∀ (i : ι), (X i).Nonempty
    hXdisj : Pairwise (Function.onFun Disjoint X)
    hYdisj : Pairwise (Function.onFun Disjoint Y)
    hXYdisj : ∀ (i j : ι), Disjoint (X i) (Y j)
    hX : ∀ (i : ι), HasSubset.Subset (HSMul.hSMul (a i) (HasCompl.compl (Y i))) (X …
    hY : ∀ (i : ι), HasSubset.Subset (HSMul.hSMul (Inv.inv a i) (HasCompl.compl (X …
    H : ι → Type := fun _i => FreeGroup Unit
    f : (i : ι) → MonoidHom (H i) G := fun i => FreeGroup.lift fun x => a i
    ⊢ Function.Injective ⇑(Monoid.CoprodI.lift fun i => FreeGroup.lift fun x => a i)
  -/
  let X' : ι → Set α := fun i => X i ∪ Y i
  /-
    ι : Type u_1
    inst✝² : Nontrivial ι
    G : Type u_1
    inst✝¹ : Group G
    a : ι → G
    α : Type u_4
    inst✝ : MulAction G α
    X Y : ι → Set α
    hXnonempty : ∀ (i : ι), (X i).Nonempty
    hXdisj : Pairwise (Function.onFun Disjoint X)
    hYdisj : Pairwise (Function.onFun Disjoint Y)
    hXYdisj : ∀ (i j : ι), Disjoint (X i) (Y j)
    hX : ∀ (i : ι), HasSubset.Subset (HSMul.hSMul (a i) (HasCompl.compl (Y i))) (X …
    hY : ∀ (i : ι), HasSubset.Subset (HSMul.hSMul (Inv.inv a i) (HasCompl.compl (X …
    H : ι → Type := fun _i => FreeGroup Unit
    f : (i : ι) → MonoidHom (H i) G := fun i => FreeGroup.lift fun x => a i
    X' : ι → Set α := fun i => Union.union (X i) (Y i)
    ⊢ Function.Injective ⇑(Monoid.CoprodI.lift fun i => FreeGroup.lift fun x => a i)
  -/
  apply lift_injective_of_ping_pong f _ X'
    /-
      case hXnonempty
      ι : Type u_1
      inst✝² : Nontrivial ι
      G : Type u_1
      inst✝¹ : Group G
      a : ι → G
      α : Type u_4
      inst✝ : MulAction G α
      X Y : ι → Set α
      hXnonempty : ∀ (i : ι), (X i).Nonempty
      hXdisj : Pairwise (Function.onFun Disjoint X)
      hYdisj : Pairwise (Function.onFun Disjoint Y)
      hXYdisj : ∀ (i j : ι), Disjoint (X i) (Y j)
      hX : ∀ (i : ι), HasSubset.Subset (HSMul.hSMul (a i) (HasCompl.compl (Y i))) (X …
      hY : ∀ (i : ι), HasSubset.Subset (HSMul.hSMul (Inv.inv a i) (HasCompl.compl (X …
      H : ι → Type := fun _i => FreeGroup Unit
      f : (i : ι) → MonoidHom (H i) G := fun i => FreeGroup.lift fun x => a i
      X' : ι → Set α := fun i => Union.union (X i) (Y i)
      ⊢ ∀ (i : ι), (X' i).Nonempty
    -/
  · show ∀ i, (X' i).Nonempty
    /-
      case hXnonempty
      ι : Type u_1
      inst✝² : Nontrivial ι
      G : Type u_1
      inst✝¹ : Group G
      a : ι → G
      α : Type u_4
      inst✝ : MulAction G α
      X Y : ι → Set α
      hXnonempty : ∀ (i : ι), (X i).Nonempty
      hXdisj : Pairwise (Function.onFun Disjoint X)
      hYdisj : Pairwise (Function.onFun Disjoint Y)
      hXYdisj : ∀ (i j : ι), Disjoint (X i) (Y j)
      hX : ∀ (i : ι), HasSubset.Subset (HSMul.hSMul (a i) (HasCompl.compl (Y i))) (X …
      hY : ∀ (i : ι), HasSubset.Subset (HSMul.hSMul (Inv.inv a i) (HasCompl.compl (X …
      H : ι → Type := fun _i => FreeGroup Unit
      f : (i : ι) → MonoidHom (H i) G := fun i => FreeGroup.lift fun x => a i
      X' : ι → Set α := fun i => Union.union (X i) (Y i)
      ⊢ ∀ (i : ι), (X' i).Nonempty
    -/
    exact fun i => Set.Nonempty.inl (hXnonempty i)
    /-
      🎉 no goals
    -/
    /-
      case hXdisj
      ι : Type u_1
      inst✝² : Nontrivial ι
      G : Type u_1
      inst✝¹ : Group G
      a : ι → G
      α : Type u_4
      inst✝ : MulAction G α
      X Y : ι → Set α
      hXnonempty : ∀ (i : ι), (X i).Nonempty
      hXdisj : Pairwise (Function.onFun Disjoint X)
      hYdisj : Pairwise (Function.onFun Disjoint Y)
      hXYdisj : ∀ (i j : ι), Disjoint (X i) (Y j)
      hX : ∀ (i : ι), HasSubset.Subset (HSMul.hSMul (a i) (HasCompl.compl (Y i))) (X …
      hY : ∀ (i : ι), HasSubset.Subset (HSMul.hSMul (Inv.inv a i) (HasCompl.compl (X …
      H : ι → Type := fun _i => FreeGroup Unit
      f : (i : ι) → MonoidHom (H i) G := fun i => FreeGroup.lift fun x => a i
      X' : ι → Set α := fun i => Union.union (X i) (Y i)
      ⊢ Pairwise (Function.onFun Disjoint X')
    -/
  · show Pairwise (Disjoint on X')
    /-
      case hXdisj
      ι : Type u_1
      inst✝² : Nontrivial ι
      G : Type u_1
      inst✝¹ : Group G
      a : ι → G
      α : Type u_4
      inst✝ : MulAction G α
      X Y : ι → Set α
      hXnonempty : ∀ (i : ι), (X i).Nonempty
      hXdisj : Pairwise (Function.onFun Disjoint X)
      hYdisj : Pairwise (Function.onFun Disjoint Y)
      hXYdisj : ∀ (i j : ι), Disjoint (X i) (Y j)
      hX : ∀ (i : ι), HasSubset.Subset (HSMul.hSMul (a i) (HasCompl.compl (Y i))) (X …
      hY : ∀ (i : ι), HasSubset.Subset (HSMul.hSMul (Inv.inv a i) (HasCompl.compl (X …
      H : ι → Type := fun _i => FreeGroup Unit
      f : (i : ι) → MonoidHom (H i) G := fun i => FreeGroup.lift fun x => a i
      X' : ι → Set α := fun i => Union.union (X i) (Y i)
      ⊢ Pairwise (Function.onFun Disjoint X')
    -/
    intro i j hij
    /-
      case hXdisj
      ι : Type u_1
      inst✝² : Nontrivial ι
      G : Type u_1
      inst✝¹ : Group G
      a : ι → G
      α : Type u_4
      inst✝ : MulAction G α
      X Y : ι → Set α
      hXnonempty : ∀ (i : ι), (X i).Nonempty
      hXdisj : Pairwise (Function.onFun Disjoint X)
      hYdisj : Pairwise (Function.onFun Disjoint Y)
      hXYdisj : ∀ (i j : ι), Disjoint (X i) (Y j)
      hX : ∀ (i : ι), HasSubset.Subset (HSMul.hSMul (a i) (HasCompl.compl (Y i))) (X …
      hY : ∀ (i : ι), HasSubset.Subset (HSMul.hSMul (Inv.inv a i) (HasCompl.compl (X …
      H : ι → Type := fun _i => FreeGroup Unit
      f : (i : ι) → MonoidHom (H i) G := fun i => FreeGroup.lift fun x => a i
      X' : ι → Set α := fun i => Union.union (X i) (Y i)
      i j : ι
      hij : Ne i j
      ⊢ Function.onFun Disjoint X' i j
    -/
    simp only [X']
    /-
      case hXdisj
      ι : Type u_1
      inst✝² : Nontrivial ι
      G : Type u_1
      inst✝¹ : Group G
      a : ι → G
      α : Type u_4
      inst✝ : MulAction G α
      X Y : ι → Set α
      hXnonempty : ∀ (i : ι), (X i).Nonempty
      hXdisj : Pairwise (Function.onFun Disjoint X)
      hYdisj : Pairwise (Function.onFun Disjoint Y)
      hXYdisj : ∀ (i j : ι), Disjoint (X i) (Y j)
      hX : ∀ (i : ι), HasSubset.Subset (HSMul.hSMul (a i) (HasCompl.compl (Y i))) (X …
      hY : ∀ (i : ι), HasSubset.Subset (HSMul.hSMul (Inv.inv a i) (HasCompl.compl (X …
      H : ι → Type := fun _i => FreeGroup Unit
      f : (i : ι) → MonoidHom (H i) G := fun i => FreeGroup.lift fun x => a i
      X' : ι → Set α := fun i => Union.union (X i) (Y i)
      i j : ι
      hij : Ne i j
      ⊢ Function.onFun Disjoint (fun i => Union.union (X i) (Y i)) i j
    -/
    apply Disjoint.union_left <;> apply Disjoint.union_right
      /-
        case hXdisj.hs.ht
        ι : Type u_1
        inst✝² : Nontrivial ι
        G : Type u_1
        inst✝¹ : Group G
        a : ι → G
        α : Type u_4
        inst✝ : MulAction G α
        X Y : ι → Set α
        hXnonempty : ∀ (i : ι), (X i).Nonempty
        hXdisj : Pairwise (Function.onFun Disjoint X)
        hYdisj : Pairwise (Function.onFun Disjoint Y)
        hXYdisj : ∀ (i j : ι), Disjoint (X i) (Y j)
        hX : ∀ (i : ι), HasSubset.Subset (HSMul.hSMul (a i) (HasCompl.compl (Y i))) (X …
        hY : ∀ (i : ι), HasSubset.Subset (HSMul.hSMul (Inv.inv a i) (HasCompl.compl (X …
        H : ι → Type := fun _i => FreeGroup Unit
        f : (i : ι) → MonoidHom (H i) G := fun i => FreeGroup.lift fun x => a i
        X' : ι → Set α := fun i => Union.union (X i) (Y i)
        i j : ι
        hij : Ne i j
        ⊢ Disjoint (X i) (X j)
      -/
    · exact hXdisj hij
      /-
        🎉 no goals
      -/
      /-
        case hXdisj.hs.hu
        ι : Type u_1
        inst✝² : Nontrivial ι
        G : Type u_1
        inst✝¹ : Group G
        a : ι → G
        α : Type u_4
        inst✝ : MulAction G α
        X Y : ι → Set α
        hXnonempty : ∀ (i : ι), (X i).Nonempty
        hXdisj : Pairwise (Function.onFun Disjoint X)
        hYdisj : Pairwise (Function.onFun Disjoint Y)
        hXYdisj : ∀ (i j : ι), Disjoint (X i) (Y j)
        hX : ∀ (i : ι), HasSubset.Subset (HSMul.hSMul (a i) (HasCompl.compl (Y i))) (X …
        hY : ∀ (i : ι), HasSubset.Subset (HSMul.hSMul (Inv.inv a i) (HasCompl.compl (X …
        H : ι → Type := fun _i => FreeGroup Unit
        f : (i : ι) → MonoidHom (H i) G := fun i => FreeGroup.lift fun x => a i
        X' : ι → Set α := fun i => Union.union (X i) (Y i)
        i j : ι
        hij : Ne i j
        ⊢ Disjoint (X i) (Y j)
      -/
    · exact hXYdisj i j
      /-
        🎉 no goals
      -/
      /-
        case hXdisj.ht.ht
        ι : Type u_1
        inst✝² : Nontrivial ι
        G : Type u_1
        inst✝¹ : Group G
        a : ι → G
        α : Type u_4
        inst✝ : MulAction G α
        X Y : ι → Set α
        hXnonempty : ∀ (i : ι), (X i).Nonempty
        hXdisj : Pairwise (Function.onFun Disjoint X)
        hYdisj : Pairwise (Function.onFun Disjoint Y)
        hXYdisj : ∀ (i j : ι), Disjoint (X i) (Y j)
        hX : ∀ (i : ι), HasSubset.Subset (HSMul.hSMul (a i) (HasCompl.compl (Y i))) (X …
        hY : ∀ (i : ι), HasSubset.Subset (HSMul.hSMul (Inv.inv a i) (HasCompl.compl (X …
        H : ι → Type := fun _i => FreeGroup Unit
        f : (i : ι) → MonoidHom (H i) G := fun i => FreeGroup.lift fun x => a i
        X' : ι → Set α := fun i => Union.union (X i) (Y i)
        i j : ι
        hij : Ne i j
        ⊢ Disjoint (Y i) (X j)
      -/
    · exact (hXYdisj j i).symm
      /-
        🎉 no goals
      -/
      /-
        case hXdisj.ht.hu
        ι : Type u_1
        inst✝² : Nontrivial ι
        G : Type u_1
        inst✝¹ : Group G
        a : ι → G
        α : Type u_4
        inst✝ : MulAction G α
        X Y : ι → Set α
        hXnonempty : ∀ (i : ι), (X i).Nonempty
        hXdisj : Pairwise (Function.onFun Disjoint X)
        hYdisj : Pairwise (Function.onFun Disjoint Y)
        hXYdisj : ∀ (i j : ι), Disjoint (X i) (Y j)
        hX : ∀ (i : ι), HasSubset.Subset (HSMul.hSMul (a i) (HasCompl.compl (Y i))) (X …
        hY : ∀ (i : ι), HasSubset.Subset (HSMul.hSMul (Inv.inv a i) (HasCompl.compl (X …
        H : ι → Type := fun _i => FreeGroup Unit
        f : (i : ι) → MonoidHom (H i) G := fun i => FreeGroup.lift fun x => a i
        X' : ι → Set α := fun i => Union.union (X i) (Y i)
        i j : ι
        hij : Ne i j
        ⊢ Disjoint (Y i) (Y j)
      -/
    · exact hYdisj hij
      /-
        🎉 no goals
      -/
    /-
      case hpp
      ι : Type u_1
      inst✝² : Nontrivial ι
      G : Type u_1
      inst✝¹ : Group G
      a : ι → G
      α : Type u_4
      inst✝ : MulAction G α
      X Y : ι → Set α
      hXnonempty : ∀ (i : ι), (X i).Nonempty
      hXdisj : Pairwise (Function.onFun Disjoint X)
      hYdisj : Pairwise (Function.onFun Disjoint Y)
      hXYdisj : ∀ (i j : ι), Disjoint (X i) (Y j)
      hX : ∀ (i : ι), HasSubset.Subset (HSMul.hSMul (a i) (HasCompl.compl (Y i))) (X …
      hY : ∀ (i : ι), HasSubset.Subset (HSMul.hSMul (Inv.inv a i) (HasCompl.compl (X …
      H : ι → Type := fun _i => FreeGroup Unit
      f : (i : ι) → MonoidHom (H i) G := fun i => FreeGroup.lift fun x => a i
      X' : ι → Set α := fun i => Union.union (X i) (Y i)
      ⊢ Pairwise fun i j => ∀ (h : H i), Ne h 1 → HasSubset.Subset (HSMul.hSMul ((f  …
    -/
  · show Pairwise fun i j => ∀ h : H i, h ≠ 1 → f i h • X' j ⊆ X' i
    /-
      case hpp
      ι : Type u_1
      inst✝² : Nontrivial ι
      G : Type u_1
      inst✝¹ : Group G
      a : ι → G
      α : Type u_4
      inst✝ : MulAction G α
      X Y : ι → Set α
      hXnonempty : ∀ (i : ι), (X i).Nonempty
      hXdisj : Pairwise (Function.onFun Disjoint X)
      hYdisj : Pairwise (Function.onFun Disjoint Y)
      hXYdisj : ∀ (i j : ι), Disjoint (X i) (Y j)
      hX : ∀ (i : ι), HasSubset.Subset (HSMul.hSMul (a i) (HasCompl.compl (Y i))) (X …
      hY : ∀ (i : ι), HasSubset.Subset (HSMul.hSMul (Inv.inv a i) (HasCompl.compl (X …
      H : ι → Type := fun _i => FreeGroup Unit
      f : (i : ι) → MonoidHom (H i) G := fun i => FreeGroup.lift fun x => a i
      X' : ι → Set α := fun i => Union.union (X i) (Y i)
      ⊢ Pairwise fun i j => ∀ (h : H i), Ne h 1 → HasSubset.Subset (HSMul.hSMul ((f  …
    -/
    rintro i j hij
    -- use free_group unit ≃ ℤ
    /-
      case hpp
      ι : Type u_1
      inst✝² : Nontrivial ι
      G : Type u_1
      inst✝¹ : Group G
      a : ι → G
      α : Type u_4
      inst✝ : MulAction G α
      X Y : ι → Set α
      hXnonempty : ∀ (i : ι), (X i).Nonempty
      hXdisj : Pairwise (Function.onFun Disjoint X)
      hYdisj : Pairwise (Function.onFun Disjoint Y)
      hXYdisj : ∀ (i j : ι), Disjoint (X i) (Y j)
      hX : ∀ (i : ι), HasSubset.Subset (HSMul.hSMul (a i) (HasCompl.compl (Y i))) (X …
      hY : ∀ (i : ι), HasSubset.Subset (HSMul.hSMul (Inv.inv a i) (HasCompl.compl (X …
      H : ι → Type := fun _i => FreeGroup Unit
      f : (i : ι) → MonoidHom (H i) G := fun i => FreeGroup.lift fun x => a i
      X' : ι → Set α := fun i => Union.union (X i) (Y i)
      i j : ι
      hij : Ne i j
      ⊢ ∀ (h : H i), Ne h 1 → HasSubset.Subset (HSMul.hSMul ((f i) h) (X' j)) (X' i)
    -/
    refine FreeGroup.freeGroupUnitEquivInt.forall_congr_left.mpr ?_
    /-
      case hpp
      ι : Type u_1
      inst✝² : Nontrivial ι
      G : Type u_1
      inst✝¹ : Group G
      a : ι → G
      α : Type u_4
      inst✝ : MulAction G α
      X Y : ι → Set α
      hXnonempty : ∀ (i : ι), (X i).Nonempty
      hXdisj : Pairwise (Function.onFun Disjoint X)
      hYdisj : Pairwise (Function.onFun Disjoint Y)
      hXYdisj : ∀ (i j : ι), Disjoint (X i) (Y j)
      hX : ∀ (i : ι), HasSubset.Subset (HSMul.hSMul (a i) (HasCompl.compl (Y i))) (X …
      hY : ∀ (i : ι), HasSubset.Subset (HSMul.hSMul (Inv.inv a i) (HasCompl.compl (X …
      H : ι → Type := fun _i => FreeGroup Unit
      f : (i : ι) → MonoidHom (H i) G := fun i => FreeGroup.lift fun x => a i
      X' : ι → Set α := fun i => Union.union (X i) (Y i)
      i j : ι
      hij : Ne i j
      ⊢ ∀ (b : Int), Ne (FreeGroup.freeGroupUnitEquivInt.symm b) 1 → HasSubset.Subse …
    -/
    intro n hne1
    /-
      case hpp
      ι : Type u_1
      inst✝² : Nontrivial ι
      G : Type u_1
      inst✝¹ : Group G
      a : ι → G
      α : Type u_4
      inst✝ : MulAction G α
      X Y : ι → Set α
      hXnonempty : ∀ (i : ι), (X i).Nonempty
      hXdisj : Pairwise (Function.onFun Disjoint X)
      hYdisj : Pairwise (Function.onFun Disjoint Y)
      hXYdisj : ∀ (i j : ι), Disjoint (X i) (Y j)
      hX : ∀ (i : ι), HasSubset.Subset (HSMul.hSMul (a i) (HasCompl.compl (Y i))) (X …
      hY : ∀ (i : ι), HasSubset.Subset (HSMul.hSMul (Inv.inv a i) (HasCompl.compl (X …
      H : ι → Type := fun _i => FreeGroup Unit
      f : (i : ι) → MonoidHom (H i) G := fun i => FreeGroup.lift fun x => a i
      X' : ι → Set α := fun i => Union.union (X i) (Y i)
      i j : ι
      hij : Ne i j
      n : Int
      hne1 : Ne (FreeGroup.freeGroupUnitEquivInt.symm n) 1
      ⊢ HasSubset.Subset (HSMul.hSMul ((f i) (FreeGroup.freeGroupUnitEquivInt.symm n …
    -/
    change FreeGroup.lift (fun _ => a i) (FreeGroup.of () ^ n) • X' j ⊆ X' i
    /-
      case hpp
      ι : Type u_1
      inst✝² : Nontrivial ι
      G : Type u_1
      inst✝¹ : Group G
      a : ι → G
      α : Type u_4
      inst✝ : MulAction G α
      X Y : ι → Set α
      hXnonempty : ∀ (i : ι), (X i).Nonempty
      hXdisj : Pairwise (Function.onFun Disjoint X)
      hYdisj : Pairwise (Function.onFun Disjoint Y)
      hXYdisj : ∀ (i j : ι), Disjoint (X i) (Y j)
      hX : ∀ (i : ι), HasSubset.Subset (HSMul.hSMul (a i) (HasCompl.compl (Y i))) (X …
      hY : ∀ (i : ι), HasSubset.Subset (HSMul.hSMul (Inv.inv a i) (HasCompl.compl (X …
      H : ι → Type := fun _i => FreeGroup Unit
      f : (i : ι) → MonoidHom (H i) G := fun i => FreeGroup.lift fun x => a i
      X' : ι → Set α := fun i => Union.union (X i) (Y i)
      i j : ι
      hij : Ne i j
      n : Int
      hne1 : Ne (FreeGroup.freeGroupUnitEquivInt.symm n) 1
      ⊢ HasSubset.Subset (HSMul.hSMul ((FreeGroup.lift fun x => a i) (HPow.hPow (Fre …
    -/
    simp only [map_zpow, FreeGroup.lift.of]
    /-
      case hpp
      ι : Type u_1
      inst✝² : Nontrivial ι
      G : Type u_1
      inst✝¹ : Group G
      a : ι → G
      α : Type u_4
      inst✝ : MulAction G α
      X Y : ι → Set α
      hXnonempty : ∀ (i : ι), (X i).Nonempty
      hXdisj : Pairwise (Function.onFun Disjoint X)
      hYdisj : Pairwise (Function.onFun Disjoint Y)
      hXYdisj : ∀ (i j : ι), Disjoint (X i) (Y j)
      hX : ∀ (i : ι), HasSubset.Subset (HSMul.hSMul (a i) (HasCompl.compl (Y i))) (X …
      hY : ∀ (i : ι), HasSubset.Subset (HSMul.hSMul (Inv.inv a i) (HasCompl.compl (X …
      H : ι → Type := fun _i => FreeGroup Unit
      f : (i : ι) → MonoidHom (H i) G := fun i => FreeGroup.lift fun x => a i
      X' : ι → Set α := fun i => Union.union (X i) (Y i)
      i j : ι
      hij : Ne i j
      n : Int
      hne1 : Ne (FreeGroup.freeGroupUnitEquivInt.symm n) 1
      ⊢ HasSubset.Subset (HSMul.hSMul (HPow.hPow (a i) n) (X' j)) (X' i)
    -/
    change a i ^ n • X' j ⊆ X' i
    have hnne0 : n ≠ 0 := by
      rintro rfl
      apply hne1
      simp [H, FreeGroup.freeGroupUnitEquivInt]
    /-
      case hpp
      ι : Type u_1
      inst✝² : Nontrivial ι
      G : Type u_1
      inst✝¹ : Group G
      a : ι → G
      α : Type u_4
      inst✝ : MulAction G α
      X Y : ι → Set α
      hXnonempty : ∀ (i : ι), (X i).Nonempty
      hXdisj : Pairwise (Function.onFun Disjoint X)
      hYdisj : Pairwise (Function.onFun Disjoint Y)
      hXYdisj : ∀ (i j : ι), Disjoint (X i) (Y j)
      hX : ∀ (i : ι), HasSubset.Subset (HSMul.hSMul (a i) (HasCompl.compl (Y i))) (X …
      hY : ∀ (i : ι), HasSubset.Subset (HSMul.hSMul (Inv.inv a i) (HasCompl.compl (X …
      H : ι → Type := fun _i => FreeGroup Unit
      f : (i : ι) → MonoidHom (H i) G := fun i => FreeGroup.lift fun x => a i
      X' : ι → Set α := fun i => Union.union (X i) (Y i)
      i j : ι
      hij : Ne i j
      n : Int
      hne1 : Ne (FreeGroup.freeGroupUnitEquivInt.symm n) 1
      hnne0 : Ne n 0
      ⊢ HasSubset.Subset (HSMul.hSMul (HPow.hPow (a i) n) (X' j)) (X' i)
    -/
    clear hne1
    /-
      case hpp
      ι : Type u_1
      inst✝² : Nontrivial ι
      G : Type u_1
      inst✝¹ : Group G
      a : ι → G
      α : Type u_4
      inst✝ : MulAction G α
      X Y : ι → Set α
      hXnonempty : ∀ (i : ι), (X i).Nonempty
      hXdisj : Pairwise (Function.onFun Disjoint X)
      hYdisj : Pairwise (Function.onFun Disjoint Y)
      hXYdisj : ∀ (i j : ι), Disjoint (X i) (Y j)
      hX : ∀ (i : ι), HasSubset.Subset (HSMul.hSMul (a i) (HasCompl.compl (Y i))) (X …
      hY : ∀ (i : ι), HasSubset.Subset (HSMul.hSMul (Inv.inv a i) (HasCompl.compl (X …
      H : ι → Type := fun _i => FreeGroup Unit
      f : (i : ι) → MonoidHom (H i) G := fun i => FreeGroup.lift fun x => a i
      X' : ι → Set α := fun i => Union.union (X i) (Y i)
      i j : ι
      hij : Ne i j
      n : Int
      hnne0 : Ne n 0
      ⊢ HasSubset.Subset (HSMul.hSMul (HPow.hPow (a i) n) (X' j)) (X' i)
    -/
    simp only [X']
    -- Positive and negative powers separately
    /-
      case hpp
      ι : Type u_1
      inst✝² : Nontrivial ι
      G : Type u_1
      inst✝¹ : Group G
      a : ι → G
      α : Type u_4
      inst✝ : MulAction G α
      X Y : ι → Set α
      hXnonempty : ∀ (i : ι), (X i).Nonempty
      hXdisj : Pairwise (Function.onFun Disjoint X)
      hYdisj : Pairwise (Function.onFun Disjoint Y)
      hXYdisj : ∀ (i j : ι), Disjoint (X i) (Y j)
      hX : ∀ (i : ι), HasSubset.Subset (HSMul.hSMul (a i) (HasCompl.compl (Y i))) (X …
      hY : ∀ (i : ι), HasSubset.Subset (HSMul.hSMul (Inv.inv a i) (HasCompl.compl (X …
      H : ι → Type := fun _i => FreeGroup Unit
      f : (i : ι) → MonoidHom (H i) G := fun i => FreeGroup.lift fun x => a i
      X' : ι → Set α := fun i => Union.union (X i) (Y i)
      i j : ι
      hij : Ne i j
      n : Int
      hnne0 : Ne n 0
      ⊢ HasSubset.Subset (HSMul.hSMul (HPow.hPow (a i) n) (Union.union (X j) (Y j))) …
    -/
    cases' (lt_or_gt_of_ne hnne0).symm with hlt hgt
      /-
        case hpp.inl
        ι : Type u_1
        inst✝² : Nontrivial ι
        G : Type u_1
        inst✝¹ : Group G
        a : ι → G
        α : Type u_4
        inst✝ : MulAction G α
        X Y : ι → Set α
        hXnonempty : ∀ (i : ι), (X i).Nonempty
        hXdisj : Pairwise (Function.onFun Disjoint X)
        hYdisj : Pairwise (Function.onFun Disjoint Y)
        hXYdisj : ∀ (i j : ι), Disjoint (X i) (Y j)
        hX : ∀ (i : ι), HasSubset.Subset (HSMul.hSMul (a i) (HasCompl.compl (Y i))) (X …
        hY : ∀ (i : ι), HasSubset.Subset (HSMul.hSMul (Inv.inv a i) (HasCompl.compl (X …
        H : ι → Type := fun _i => FreeGroup Unit
        f : (i : ι) → MonoidHom (H i) G := fun i => FreeGroup.lift fun x => a i
        X' : ι → Set α := fun i => Union.union (X i) (Y i)
        i j : ι
        hij : Ne i j
        n : Int
        hnne0 : Ne n 0
        hlt : GT.gt n 0
        ⊢ HasSubset.Subset (HSMul.hSMul (HPow.hPow (a i) n) (Union.union (X j) (Y j))) …
      -/
    · have h1n : 1 ≤ n := hlt
      calc
        a i ^ n • X' j ⊆ a i ^ n • (Y i)ᶜ :=
          smul_set_mono ((hXYdisj j i).union_left <| hYdisj hij.symm).subset_compl_right
        _ ⊆ X i := by
          clear hnne0 hlt
          refine Int.le_induction (P := fun n => a i ^ n • (Y i)ᶜ ⊆ X i) ?_ ?_ n h1n
          · dsimp
            rw [zpow_one]
            exact hX i
          · dsimp
            intro n _hle hi
            calc
              a i ^ (n + 1) • (Y i)ᶜ = (a i ^ n * a i) • (Y i)ᶜ := by rw [zpow_add, zpow_one]
              _ = a i ^ n • a i • (Y i)ᶜ := MulAction.mul_smul _ _ _
              _ ⊆ a i ^ n • X i := smul_set_mono <| hX i
              _ ⊆ a i ^ n • (Y i)ᶜ := smul_set_mono (hXYdisj i i).subset_compl_right
              _ ⊆ X i := hi
        _ ⊆ X' i := Set.subset_union_left
    · have h1n : n ≤ -1 := by
        apply Int.le_of_lt_add_one
        simpa using hgt
      calc
        a i ^ n • X' j ⊆ a i ^ n • (X i)ᶜ :=
          smul_set_mono ((hXdisj hij.symm).union_left (hXYdisj i j).symm).subset_compl_right
        _ ⊆ Y i := by
          refine Int.le_induction_down (P := fun n => a i ^ n • (X i)ᶜ ⊆ Y i) ?_ ?_ _ h1n
          · dsimp
            rw [zpow_neg, zpow_one]
            exact hY i
          · dsimp
            intro n _ hi
            calc
              a i ^ (n - 1) • (X i)ᶜ = (a i ^ n * (a i)⁻¹) • (X i)ᶜ := by rw [zpow_sub, zpow_one]
              _ = a i ^ n • (a i)⁻¹ • (X i)ᶜ := MulAction.mul_smul _ _ _
              _ ⊆ a i ^ n • Y i := smul_set_mono <| hY i
              _ ⊆ a i ^ n • (X i)ᶜ := smul_set_mono (hXYdisj i i).symm.subset_compl_right
              _ ⊆ Y i := hi
        _ ⊆ X' i := Set.subset_union_right
  /-
    ι : Type u_1
    inst✝² : Nontrivial ι
    G : Type u_1
    inst✝¹ : Group G
    a : ι → G
    α : Type u_4
    inst✝ : MulAction G α
    X Y : ι → Set α
    hXnonempty : ∀ (i : ι), (X i).Nonempty
    hXdisj : Pairwise (Function.onFun Disjoint X)
    hYdisj : Pairwise (Function.onFun Disjoint Y)
    hXYdisj : ∀ (i j : ι), Disjoint (X i) (Y j)
    hX : ∀ (i : ι), HasSubset.Subset (HSMul.hSMul (a i) (HasCompl.compl (Y i))) (X …
    hY : ∀ (i : ι), HasSubset.Subset (HSMul.hSMul (Inv.inv a i) (HasCompl.compl (X …
    H : ι → Type := fun _i => FreeGroup Unit
    f : (i : ι) → MonoidHom (H i) G := fun i => FreeGroup.lift fun x => a i
    X' : ι → Set α := fun i => Union.union (X i) (Y i)
    ⊢ Or (LE.le 3 (Cardinal.mk ι)) (Exists fun i => LE.le 3 (Cardinal.mk (H i)))
  -/
  show _ ∨ ∃ i, 3 ≤ #(H i)
  /-
    ι : Type u_1
    inst✝² : Nontrivial ι
    G : Type u_1
    inst✝¹ : Group G
    a : ι → G
    α : Type u_4
    inst✝ : MulAction G α
    X Y : ι → Set α
    hXnonempty : ∀ (i : ι), (X i).Nonempty
    hXdisj : Pairwise (Function.onFun Disjoint X)
    hYdisj : Pairwise (Function.onFun Disjoint Y)
    hXYdisj : ∀ (i j : ι), Disjoint (X i) (Y j)
    hX : ∀ (i : ι), HasSubset.Subset (HSMul.hSMul (a i) (HasCompl.compl (Y i))) (X …
    hY : ∀ (i : ι), HasSubset.Subset (HSMul.hSMul (Inv.inv a i) (HasCompl.compl (X …
    H : ι → Type := fun _i => FreeGroup Unit
    f : (i : ι) → MonoidHom (H i) G := fun i => FreeGroup.lift fun x => a i
    X' : ι → Set α := fun i => Union.union (X i) (Y i)
    ⊢ Or (LE.le 3 (Cardinal.mk ι)) (Exists fun i => LE.le 3 (Cardinal.mk (H i)))
  -/
  inhabit ι
  /-
    ι : Type u_1
    inst✝² : Nontrivial ι
    G : Type u_1
    inst✝¹ : Group G
    a : ι → G
    α : Type u_4
    inst✝ : MulAction G α
    X Y : ι → Set α
    hXnonempty : ∀ (i : ι), (X i).Nonempty
    hXdisj : Pairwise (Function.onFun Disjoint X)
    hYdisj : Pairwise (Function.onFun Disjoint Y)
    hXYdisj : ∀ (i j : ι), Disjoint (X i) (Y j)
    hX : ∀ (i : ι), HasSubset.Subset (HSMul.hSMul (a i) (HasCompl.compl (Y i))) (X …
    hY : ∀ (i : ι), HasSubset.Subset (HSMul.hSMul (Inv.inv a i) (HasCompl.compl (X …
    H : ι → Type := fun _i => FreeGroup Unit
    f : (i : ι) → MonoidHom (H i) G := fun i => FreeGroup.lift fun x => a i
    X' : ι → Set α := fun i => Union.union (X i) (Y i)
    inhabited_h : Inhabited ι
    ⊢ Or (LE.le 3 (Cardinal.mk ι)) (Exists fun i => LE.le 3 (Cardinal.mk (H i)))
  -/
  right
  /-
    case h
    ι : Type u_1
    inst✝² : Nontrivial ι
    G : Type u_1
    inst✝¹ : Group G
    a : ι → G
    α : Type u_4
    inst✝ : MulAction G α
    X Y : ι → Set α
    hXnonempty : ∀ (i : ι), (X i).Nonempty
    hXdisj : Pairwise (Function.onFun Disjoint X)
    hYdisj : Pairwise (Function.onFun Disjoint Y)
    hXYdisj : ∀ (i j : ι), Disjoint (X i) (Y j)
    hX : ∀ (i : ι), HasSubset.Subset (HSMul.hSMul (a i) (HasCompl.compl (Y i))) (X …
    hY : ∀ (i : ι), HasSubset.Subset (HSMul.hSMul (Inv.inv a i) (HasCompl.compl (X …
    H : ι → Type := fun _i => FreeGroup Unit
    f : (i : ι) → MonoidHom (H i) G := fun i => FreeGroup.lift fun x => a i
    X' : ι → Set α := fun i => Union.union (X i) (Y i)
    inhabited_h : Inhabited ι
    ⊢ Exists fun i => LE.le 3 (Cardinal.mk (H i))
  -/
  use Inhabited.default
  /-
    case h
    ι : Type u_1
    inst✝² : Nontrivial ι
    G : Type u_1
    inst✝¹ : Group G
    a : ι → G
    α : Type u_4
    inst✝ : MulAction G α
    X Y : ι → Set α
    hXnonempty : ∀ (i : ι), (X i).Nonempty
    hXdisj : Pairwise (Function.onFun Disjoint X)
    hYdisj : Pairwise (Function.onFun Disjoint Y)
    hXYdisj : ∀ (i j : ι), Disjoint (X i) (Y j)
    hX : ∀ (i : ι), HasSubset.Subset (HSMul.hSMul (a i) (HasCompl.compl (Y i))) (X …
    hY : ∀ (i : ι), HasSubset.Subset (HSMul.hSMul (Inv.inv a i) (HasCompl.compl (X …
    H : ι → Type := fun _i => FreeGroup Unit
    f : (i : ι) → MonoidHom (H i) G := fun i => FreeGroup.lift fun x => a i
    X' : ι → Set α := fun i => Union.union (X i) (Y i)
    inhabited_h : Inhabited ι
    ⊢ LE.le 3 (Cardinal.mk (H Inhabited.default))
  -/
  simp only [H]
  /-
    case h
    ι : Type u_1
    inst✝² : Nontrivial ι
    G : Type u_1
    inst✝¹ : Group G
    a : ι → G
    α : Type u_4
    inst✝ : MulAction G α
    X Y : ι → Set α
    hXnonempty : ∀ (i : ι), (X i).Nonempty
    hXdisj : Pairwise (Function.onFun Disjoint X)
    hYdisj : Pairwise (Function.onFun Disjoint Y)
    hXYdisj : ∀ (i j : ι), Disjoint (X i) (Y j)
    hX : ∀ (i : ι), HasSubset.Subset (HSMul.hSMul (a i) (HasCompl.compl (Y i))) (X …
    hY : ∀ (i : ι), HasSubset.Subset (HSMul.hSMul (Inv.inv a i) (HasCompl.compl (X …
    H : ι → Type := fun _i => FreeGroup Unit
    f : (i : ι) → MonoidHom (H i) G := fun i => FreeGroup.lift fun x => a i
    X' : ι → Set α := fun i => Union.union (X i) (Y i)
    inhabited_h : Inhabited ι
    ⊢ LE.le 3 (Cardinal.mk (FreeGroup Unit))
  -/
  rw [FreeGroup.freeGroupUnitEquivInt.cardinal_eq, Cardinal.mk_denumerable]
  /-
    case h
    ι : Type u_1
    inst✝² : Nontrivial ι
    G : Type u_1
    inst✝¹ : Group G
    a : ι → G
    α : Type u_4
    inst✝ : MulAction G α
    X Y : ι → Set α
    hXnonempty : ∀ (i : ι), (X i).Nonempty
    hXdisj : Pairwise (Function.onFun Disjoint X)
    hYdisj : Pairwise (Function.onFun Disjoint Y)
    hXYdisj : ∀ (i j : ι), Disjoint (X i) (Y j)
    hX : ∀ (i : ι), HasSubset.Subset (HSMul.hSMul (a i) (HasCompl.compl (Y i))) (X …
    hY : ∀ (i : ι), HasSubset.Subset (HSMul.hSMul (Inv.inv a i) (HasCompl.compl (X …
    H : ι → Type := fun _i => FreeGroup Unit
    f : (i : ι) → MonoidHom (H i) G := fun i => FreeGroup.lift fun x => a i
    X' : ι → Set α := fun i => Union.union (X i) (Y i)
    inhabited_h : Inhabited ι
    ⊢ LE.le 3 Cardinal.aleph0
  -/
  apply le_of_lt
  /-
    case h.hab
    ι : Type u_1
    inst✝² : Nontrivial ι
    G : Type u_1
    inst✝¹ : Group G
    a : ι → G
    α : Type u_4
    inst✝ : MulAction G α
    X Y : ι → Set α
    hXnonempty : ∀ (i : ι), (X i).Nonempty
    hXdisj : Pairwise (Function.onFun Disjoint X)
    hYdisj : Pairwise (Function.onFun Disjoint Y)
    hXYdisj : ∀ (i j : ι), Disjoint (X i) (Y j)
    hX : ∀ (i : ι), HasSubset.Subset (HSMul.hSMul (a i) (HasCompl.compl (Y i))) (X …
    hY : ∀ (i : ι), HasSubset.Subset (HSMul.hSMul (Inv.inv a i) (HasCompl.compl (X …
    H : ι → Type := fun _i => FreeGroup Unit
    f : (i : ι) → MonoidHom (H i) G := fun i => FreeGroup.lift fun x => a i
    X' : ι → Set α := fun i => Union.union (X i) (Y i)
    inhabited_h : Inhabited ι
    ⊢ LT.lt 3 Cardinal.aleph0
  -/
  exact nat_lt_aleph0 3
  /-
    🎉 no goals
  -/


