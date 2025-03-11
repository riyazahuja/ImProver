alias DirectedSystem.map_self := DirectedSystem.map_self'

alias DirectedSystem.map_map := DirectedSystem.map_map'


/-- The relation on the direct sum that generates the additive congruence that defines the
colimit as a quotient. -/
inductive DirectLimit.Eqv : DirectSum ι G → DirectSum ι G → Prop
  | of_map {i j} (h : i ≤ j) (x : G i) :
    Eqv (DirectSum.lof R ι G i x) (DirectSum.lof R ι G j <| f i j h x)


/-- The direct limit of a directed system is the modules glued together along the maps. -/
def DirectLimit [DecidableEq ι] : Type _ := (addConGen <| DirectLimit.Eqv f).Quotient


instance addCommMonoid : AddCommMonoid (DirectLimit G f) :=
  AddCon.addCommMonoid _


instance module : Module R (DirectLimit G f) where
  smul r := AddCon.lift _ ((AddCon.mk' _).comp <| smulAddHom R _ r) <|
    AddCon.addConGen_le fun x y ⟨_, _⟩ ↦ (AddCon.eq _).mpr <| by
      /-
        R : Type u_1
        inst✝⁴ : Semiring R
        ι : Type u_2
        inst✝³ : Preorder ι
        G : ι → Type u_3
        inst✝² : DecidableEq ι
        inst✝¹ : (i : ι) → AddCommMonoid (G i)
        inst✝ : (i : ι) → Module R (G i)
        f : (i j : ι) → LE.le i j → LinearMap (RingHom.id R) (G i) (G j)
        r : R
        x y : DirectSum ι G
        x✝¹ : Module.DirectLimit.Eqv f x y
        i✝ j✝ : ι
        h✝ : LE.le i✝ j✝
        x✝ : G i✝
        ⊢ (addConGen (Module.DirectLimit.Eqv f)) (((smulAddHom R (DirectSum ι G)) r) ( …
      -/
      simpa only [smulAddHom_apply, ← map_smul] using .of _ _ (.of_map _ _)
      /-
        🎉 no goals
      -/
                 /-
                   R : Type u_1
                   inst✝⁴ : Semiring R
                   ι : Type u_2
                   inst✝³ : Preorder ι
                   G : ι → Type u_3
                   inst✝² : DecidableEq ι
                   inst✝¹ : (i : ι) → AddCommMonoid (G i)
                   inst✝ : (i : ι) → Module R (G i)
                   f : (i j : ι) → LE.le i j → LinearMap (RingHom.id R) (G i) (G j)
                   ⊢ ∀ (b : Module.DirectLimit G f), Eq (HSMul.hSMul 1 b) b
                 -/
  one_smul := by rintro ⟨⟩; exact congr_arg _ (one_smul _ _)
                            /-
                              🎉 no goals
                            -/
                     /-
                       R : Type u_1
                       inst✝⁴ : Semiring R
                       ι : Type u_2
                       inst✝³ : Preorder ι
                       G : ι → Type u_3
                       inst✝² : DecidableEq ι
                       inst✝¹ : (i : ι) → AddCommMonoid (G i)
                       inst✝ : (i : ι) → Module R (G i)
                       f : (i j : ι) → LE.le i j → LinearMap (RingHom.id R) (G i) (G j)
                       x✝¹ x✝ : R
                       ⊢ ∀ (b : Module.DirectLimit G f), Eq (HSMul.hSMul (HMul.hMul x✝¹ x✝) b) (HSMul …
                     -/
  mul_smul _ _ := by rintro ⟨⟩; exact congr_arg _ (mul_smul _ _ _)
                                /-
                                  🎉 no goals
                                -/
  smul_zero _ := congr_arg _ (smul_zero _)
                   /-
                     R : Type u_1
                     inst✝⁴ : Semiring R
                     ι : Type u_2
                     inst✝³ : Preorder ι
                     G : ι → Type u_3
                     inst✝² : DecidableEq ι
                     inst✝¹ : (i : ι) → AddCommMonoid (G i)
                     inst✝ : (i : ι) → Module R (G i)
                     f : (i j : ι) → LE.le i j → LinearMap (RingHom.id R) (G i) (G j)
                     x✝ : R
                     ⊢ ∀ (x y : Module.DirectLimit G f), Eq (HSMul.hSMul x✝ (HAdd.hAdd x y)) (HAdd. …
                   -/
  smul_add _ := by rintro ⟨⟩ ⟨⟩; exact congr_arg _ (smul_add _ _ _)
                                 /-
                                   🎉 no goals
                                 -/
                     /-
                       R : Type u_1
                       inst✝⁴ : Semiring R
                       ι : Type u_2
                       inst✝³ : Preorder ι
                       G : ι → Type u_3
                       inst✝² : DecidableEq ι
                       inst✝¹ : (i : ι) → AddCommMonoid (G i)
                       inst✝ : (i : ι) → Module R (G i)
                       f : (i j : ι) → LE.le i j → LinearMap (RingHom.id R) (G i) (G j)
                       x✝¹ x✝ : R
                       ⊢ ∀ (x : Module.DirectLimit G f), Eq (HSMul.hSMul (HAdd.hAdd x✝¹ x✝) x) (HAdd. …
                     -/
  add_smul _ _ := by rintro ⟨⟩; exact congr_arg _ (add_smul _ _ _)
                                /-
                                  🎉 no goals
                                -/
                  /-
                    R : Type u_1
                    inst✝⁴ : Semiring R
                    ι : Type u_2
                    inst✝³ : Preorder ι
                    G : ι → Type u_3
                    inst✝² : DecidableEq ι
                    inst✝¹ : (i : ι) → AddCommMonoid (G i)
                    inst✝ : (i : ι) → Module R (G i)
                    f : (i j : ι) → LE.le i j → LinearMap (RingHom.id R) (G i) (G j)
                    ⊢ ∀ (x : Module.DirectLimit G f), Eq (HSMul.hSMul 0 x) 0
                  -/
  zero_smul := by rintro ⟨⟩; exact congr_arg _ (zero_smul _ _)
                             /-
                               🎉 no goals
                             -/


instance addCommGroup (G : ι → Type*) [∀ i, AddCommGroup (G i)] [∀ i, Module R (G i)]
    (f : ∀ i j, i ≤ j → G i →ₗ[R] G j) : AddCommGroup (DirectLimit G f) :=
  inferInstanceAs (AddCommGroup <| AddCon.Quotient _)


instance inhabited : Inhabited (DirectLimit G f) :=
  ⟨0⟩


instance unique [IsEmpty ι] : Unique (DirectLimit G f) :=
  inferInstanceAs <| Unique (Quotient _)


/-- The canonical map from a component to the direct limit. -/
def of (i) : G i →ₗ[R] DirectLimit G f :=
  .comp { __ := AddCon.mk' _, map_smul' := fun _ _ ↦ rfl } <| DirectSum.lof R ι G i


theorem quotMk_of (i x) : Quot.mk _ (.of G i x) = of R ι G f i x := rfl


@[simp]
theorem of_f {i j hij x} : of R ι G f j (f i j hij x) = of R ι G f i x :=
  (AddCon.eq _).mpr <| .symm <| .of _ _ (.of_map _ _)


/-- Every element of the direct limit corresponds to some element in
some component of the directed system. -/
theorem exists_of [Nonempty ι] [IsDirected ι (· ≤ ·)] (z : DirectLimit G f) :
    ∃ i x, of R ι G f i x = z :=
                    /-
                      R : Type u_1
                      inst✝⁶ : Semiring R
                      ι : Type u_2
                      inst✝⁵ : Preorder ι
                      G : ι → Type u_3
                      inst✝⁴ : DecidableEq ι
                      inst✝³ : (i : ι) → AddCommMonoid (G i)
                      inst✝² : (i : ι) → Module R (G i)
                      f : (i j : ι) → LE.le i j → LinearMap (RingHom.id R) (G i) (G j)
                      inst✝¹ : Nonempty ι
                      inst✝ : IsDirected ι fun x1 x2 => LE.le x1 x2
                      z : Module.DirectLimit G f
                      ⊢ Nonempty ι
                    -/
  Nonempty.elim (by infer_instance) fun ind : ι ↦
                    /-
                      🎉 no goals
                    -/
    Quotient.inductionOn' z fun z ↦
      DirectSum.induction_on z ⟨ind, 0, LinearMap.map_zero _⟩ (fun i x ↦ ⟨i, x, rfl⟩)
        fun p q ⟨i, x, ihx⟩ ⟨j, y, ihy⟩ ↦
        let ⟨k, hik, hjk⟩ := exists_ge_ge i j
        ⟨k, f i k hik x + f j k hjk y, by
          /-
            R : Type u_1
            inst✝⁶ : Semiring R
            ι : Type u_2
            inst✝⁵ : Preorder ι
            G : ι → Type u_3
            inst✝⁴ : DecidableEq ι
            inst✝³ : (i : ι) → AddCommMonoid (G i)
            inst✝² : (i : ι) → Module R (G i)
            f : (i j : ι) → LE.le i j → LinearMap (RingHom.id R) (G i) (G j)
            inst✝¹ : Nonempty ι
            inst✝ : IsDirected ι fun x1 x2 => LE.le x1 x2
            z✝ : Module.DirectLimit G f
            ind : ι
            z : DirectSum ι G
            p q : DirectSum ι fun i => G i
            x✝¹ : Exists fun i => Exists fun x => Eq ((Module.DirectLimit.of R ι G f i) x) …
            x✝ : Exists fun i => Exists fun x => Eq ((Module.DirectLimit.of R ι G f i) x)  …
            i : ι
            x : G i
            ihx : Eq ((Module.DirectLimit.of R ι G f i) x) (Quotient.mk'' p)
            j : ι
            y : G j
            ihy : Eq ((Module.DirectLimit.of R ι G f j) y) (Quotient.mk'' q)
            k : ι
            hik : LE.le i k
            hjk : LE.le j k
            ⊢ Eq ((Module.DirectLimit.of R ι G f k) (HAdd.hAdd ((f i k hik) x) ((f j k hjk …
          -/
          rw [LinearMap.map_add, of_f, of_f, ihx, ihy]
          /-
            R : Type u_1
            inst✝⁶ : Semiring R
            ι : Type u_2
            inst✝⁵ : Preorder ι
            G : ι → Type u_3
            inst✝⁴ : DecidableEq ι
            inst✝³ : (i : ι) → AddCommMonoid (G i)
            inst✝² : (i : ι) → Module R (G i)
            f : (i j : ι) → LE.le i j → LinearMap (RingHom.id R) (G i) (G j)
            inst✝¹ : Nonempty ι
            inst✝ : IsDirected ι fun x1 x2 => LE.le x1 x2
            z✝ : Module.DirectLimit G f
            ind : ι
            z : DirectSum ι G
            p q : DirectSum ι fun i => G i
            x✝¹ : Exists fun i => Exists fun x => Eq ((Module.DirectLimit.of R ι G f i) x) …
            x✝ : Exists fun i => Exists fun x => Eq ((Module.DirectLimit.of R ι G f i) x)  …
            i : ι
            x : G i
            ihx : Eq ((Module.DirectLimit.of R ι G f i) x) (Quotient.mk'' p)
            j : ι
            y : G j
            ihy : Eq ((Module.DirectLimit.of R ι G f j) y) (Quotient.mk'' q)
            k : ι
            hik : LE.le i k
            hjk : LE.le j k
            ⊢ Eq (HAdd.hAdd (Quotient.mk'' p) (Quotient.mk'' q)) (Quotient.mk'' (HAdd.hAdd …
          -/
          rfl ⟩
          /-
            🎉 no goals
          -/


theorem exists_of₂ [Nonempty ι] [IsDirected ι (· ≤ ·)] (z w : DirectLimit G f) :
    ∃ i x y, of R ι G f i x = z ∧ of R ι G f i y = w :=
  have ⟨i, x, hx⟩ := exists_of z
  have ⟨j, y, hy⟩ := exists_of w
  have ⟨k, hik, hjk⟩ := exists_ge_ge i j
                                   /-
                                     R : Type u_1
                                     inst✝⁶ : Semiring R
                                     ι : Type u_2
                                     inst✝⁵ : Preorder ι
                                     G : ι → Type u_3
                                     inst✝⁴ : DecidableEq ι
                                     inst✝³ : (i : ι) → AddCommMonoid (G i)
                                     inst✝² : (i : ι) → Module R (G i)
                                     f : (i j : ι) → LE.le i j → LinearMap (RingHom.id R) (G i) (G j)
                                     inst✝¹ : Nonempty ι
                                     inst✝ : IsDirected ι fun x1 x2 => LE.le x1 x2
                                     z w : Module.DirectLimit G f
                                     i : ι
                                     x : G i
                                     hx : Eq ((Module.DirectLimit.of R ι G f i) x) z
                                     j : ι
                                     y : G j
                                     hy : Eq ((Module.DirectLimit.of R ι G f j) y) w
                                     k : ι
                                     hik : LE.le i k
                                     hjk : LE.le j k
                                     ⊢ Eq ((Module.DirectLimit.of R ι G f k) ((f i k hik) x)) z
                                   -/
                                   /-
                                     🎉 no goals
                                   -/
  ⟨k, f i k hik x, f j k hjk y, by rw [of_f, hx], by rw [of_f, hy]⟩
                                                     /-
                                                       🎉 no goals
                                                     -/


@[elab_as_elim]
protected theorem induction_on [Nonempty ι] [IsDirected ι (· ≤ ·)] {C : DirectLimit G f → Prop}
    (z : DirectLimit G f) (ih : ∀ i x, C (of R ι G f i x)) : C z :=
  let ⟨i, x, h⟩ := exists_of z
  h ▸ ih i x


variable (R ι G f) in
/-- The universal property of the direct limit: maps from the components to another module
that respect the directed system structure (i.e. make some diagram commute) give rise
to a unique map out of the direct limit. -/
def lift (g : ∀ i, G i →ₗ[R] P) (Hg : ∀ i j hij x, g j (f i j hij x) = g i x) :
    DirectLimit G f →ₗ[R] P where
  __ := AddCon.lift _ (DirectSum.toModule R ι P g) <|
                                            /-
                                              R : Type u_1
                                              inst✝⁶ : Semiring R
                                              ι : Type u_2
                                              inst✝⁵ : Preorder ι
                                              G : ι → Type u_3
                                              inst✝⁴ : DecidableEq ι
                                              inst✝³ : (i : ι) → AddCommMonoid (G i)
                                              inst✝² : (i : ι) → Module R (G i)
                                              f : (i j : ι) → LE.le i j → LinearMap (RingHom.id R) (G i) (G j)
                                              P : Type u_4
                                              inst✝¹ : AddCommMonoid P
                                              inst✝ : Module R P
                                              g : (i : ι) → LinearMap (RingHom.id R) (G i) P
                                              Hg : ∀ (i j : ι) (hij : LE.le i j) (x : G i), Eq ((g j) ((f i j hij) x)) ((g i …
                                              x✝³ x✝² : DirectSum ι G
                                              x✝¹ : Module.DirectLimit.Eqv f x✝³ x✝²
                                              i✝ j✝ : ι
                                              h✝ : LE.le i✝ j✝
                                              x✝ : G i✝
                                              ⊢ (AddCon.ker ↑(DirectSum.toModule R ι P g)) ((DirectSum.lof R ι G i✝) x✝) ((D …
                                            -/
    AddCon.addConGen_le fun _ _ ⟨_, _⟩ ↦ by simpa using (Hg _ _ _ _).symm
                                            /-
                                              🎉 no goals
                                            -/
                    /-
                      R : Type u_1
                      inst✝⁶ : Semiring R
                      ι : Type u_2
                      inst✝⁵ : Preorder ι
                      G : ι → Type u_3
                      inst✝⁴ : DecidableEq ι
                      inst✝³ : (i : ι) → AddCommMonoid (G i)
                      inst✝² : (i : ι) → Module R (G i)
                      f : (i j : ι) → LE.le i j → LinearMap (RingHom.id R) (G i) (G j)
                      P : Type u_4
                      inst✝¹ : AddCommMonoid P
                      inst✝ : Module R P
                      g : (i : ι) → LinearMap (RingHom.id R) (G i) P
                      Hg : ∀ (i j : ι) (hij : LE.le i j) (x : G i), Eq ((g j) ((f i j hij) x)) ((g i …
                      r : R
                      ⊢ ∀ (x : Module.DirectLimit G f), Eq ({ toFun := (↑__spread✝⁻⁰).toFun, map_add …
                    -/
  map_smul' r := by rintro ⟨x⟩; exact map_smul (DirectSum.toModule R ι P g) r x
                                /-
                                  🎉 no goals
                                -/


@[simp] theorem lift_of {i} (x) : lift R ι G f g Hg (of R ι G f i x) = g i x :=
  DirectSum.toModule_lof R _ _


theorem lift_unique (F : DirectLimit G f →ₗ[R] P) (x) :
                                                                            /-
                                                                              R : Type u_1
                                                                              inst✝⁶ : Semiring R
                                                                              ι : Type u_2
                                                                              inst✝⁵ : Preorder ι
                                                                              G : ι → Type u_3
                                                                              inst✝⁴ : DecidableEq ι
                                                                              inst✝³ : (i : ι) → AddCommMonoid (G i)
                                                                              inst✝² : (i : ι) → Module R (G i)
                                                                              f : (i j : ι) → LE.le i j → LinearMap (RingHom.id R) (G i) (G j)
                                                                              P : Type u_4
                                                                              inst✝¹ : AddCommMonoid P
                                                                              inst✝ : Module R P
                                                                              g : (i : ι) → LinearMap (RingHom.id R) (G i) P
                                                                              Hg : ∀ (i j : ι) (hij : LE.le i j) (x : G i), Eq ((g j) ((f i j hij) x)) ((g i …
                                                                              F : LinearMap (RingHom.id R) (Module.DirectLimit G f) P
                                                                              x✝ : Module.DirectLimit G f
                                                                              i j : ι
                                                                              hij : LE.le i j
                                                                              x : G i
                                                                              ⊢ Eq (((fun i => F.comp (Module.DirectLimit.of R ι G f i)) j) ((f i j hij) x)) …
                                                                            -/
    F x = lift R ι G f (fun i ↦ F.comp <| of R ι G f i) (fun i j hij x ↦ by simp) x := by
                                                                            /-
                                                                              🎉 no goals
                                                                            -/
  /-
    R : Type u_1
    inst✝⁶ : Semiring R
    ι : Type u_2
    inst✝⁵ : Preorder ι
    G : ι → Type u_3
    inst✝⁴ : DecidableEq ι
    inst✝³ : (i : ι) → AddCommMonoid (G i)
    inst✝² : (i : ι) → Module R (G i)
    f : (i j : ι) → LE.le i j → LinearMap (RingHom.id R) (G i) (G j)
    P : Type u_4
    inst✝¹ : AddCommMonoid P
    inst✝ : Module R P
    F : LinearMap (RingHom.id R) (Module.DirectLimit G f) P
    x : Module.DirectLimit G f
    ⊢ Eq (F x) ((Module.DirectLimit.lift R ι G f (fun i => F.comp (Module.DirectLi …
  -/
  rcases x with ⟨x⟩
  /-
    case mk
    R : Type u_1
    inst✝⁶ : Semiring R
    ι : Type u_2
    inst✝⁵ : Preorder ι
    G : ι → Type u_3
    inst✝⁴ : DecidableEq ι
    inst✝³ : (i : ι) → AddCommMonoid (G i)
    inst✝² : (i : ι) → Module R (G i)
    f : (i j : ι) → LE.le i j → LinearMap (RingHom.id R) (G i) (G j)
    P : Type u_4
    inst✝¹ : AddCommMonoid P
    inst✝ : Module R P
    F : LinearMap (RingHom.id R) (Module.DirectLimit G f) P
    x✝ : Module.DirectLimit G f
    x : DirectSum ι G
    ⊢ Eq (F (Quot.mk (⇑(addConGen (Module.DirectLimit.Eqv f)).toSetoid) x)) ((Modu …
  -/
  exact x.induction_on (by simp) (fun _ _ ↦ .symm <| lift_of ..) (by simp +contextual)
  /-
    🎉 no goals
  -/


lemma lift_injective [IsDirected ι (· ≤ ·)]
    (injective : ∀ i, Function.Injective <| g i) :
    Function.Injective (lift R ι G f g Hg) := by
  /-
    R : Type u_1
    inst✝⁷ : Semiring R
    ι : Type u_2
    inst✝⁶ : Preorder ι
    G : ι → Type u_3
    inst✝⁵ : DecidableEq ι
    inst✝⁴ : (i : ι) → AddCommMonoid (G i)
    inst✝³ : (i : ι) → Module R (G i)
    f : (i j : ι) → LE.le i j → LinearMap (RingHom.id R) (G i) (G j)
    P : Type u_4
    inst✝² : AddCommMonoid P
    inst✝¹ : Module R P
    g : (i : ι) → LinearMap (RingHom.id R) (G i) P
    Hg : ∀ (i j : ι) (hij : LE.le i j) (x : G i), Eq ((g j) ((f i j hij) x)) ((g i …
    inst✝ : IsDirected ι fun x1 x2 => LE.le x1 x2
    injective : ∀ (i : ι), Function.Injective ⇑(g i)
    ⊢ Function.Injective ⇑(Module.DirectLimit.lift R ι G f g Hg)
  -/
  cases isEmpty_or_nonempty ι
    /-
      case inl
      R : Type u_1
      inst✝⁷ : Semiring R
      ι : Type u_2
      inst✝⁶ : Preorder ι
      G : ι → Type u_3
      inst✝⁵ : DecidableEq ι
      inst✝⁴ : (i : ι) → AddCommMonoid (G i)
      inst✝³ : (i : ι) → Module R (G i)
      f : (i j : ι) → LE.le i j → LinearMap (RingHom.id R) (G i) (G j)
      P : Type u_4
      inst✝² : AddCommMonoid P
      inst✝¹ : Module R P
      g : (i : ι) → LinearMap (RingHom.id R) (G i) P
      Hg : ∀ (i j : ι) (hij : LE.le i j) (x : G i), Eq ((g j) ((f i j hij) x)) ((g i …
      inst✝ : IsDirected ι fun x1 x2 => LE.le x1 x2
      injective : ∀ (i : ι), Function.Injective ⇑(g i)
      h✝ : IsEmpty ι
      ⊢ Function.Injective ⇑(Module.DirectLimit.lift R ι G f g Hg)
    -/
  · apply Function.injective_of_subsingleton
    /-
      🎉 no goals
    -/
  /-
    case inr
    R : Type u_1
    inst✝⁷ : Semiring R
    ι : Type u_2
    inst✝⁶ : Preorder ι
    G : ι → Type u_3
    inst✝⁵ : DecidableEq ι
    inst✝⁴ : (i : ι) → AddCommMonoid (G i)
    inst✝³ : (i : ι) → Module R (G i)
    f : (i j : ι) → LE.le i j → LinearMap (RingHom.id R) (G i) (G j)
    P : Type u_4
    inst✝² : AddCommMonoid P
    inst✝¹ : Module R P
    g : (i : ι) → LinearMap (RingHom.id R) (G i) P
    Hg : ∀ (i j : ι) (hij : LE.le i j) (x : G i), Eq ((g j) ((f i j hij) x)) ((g i …
    inst✝ : IsDirected ι fun x1 x2 => LE.le x1 x2
    injective : ∀ (i : ι), Function.Injective ⇑(g i)
    h✝ : Nonempty ι
    ⊢ Function.Injective ⇑(Module.DirectLimit.lift R ι G f g Hg)
  -/
  intros z w eq
  /-
    case inr
    R : Type u_1
    inst✝⁷ : Semiring R
    ι : Type u_2
    inst✝⁶ : Preorder ι
    G : ι → Type u_3
    inst✝⁵ : DecidableEq ι
    inst✝⁴ : (i : ι) → AddCommMonoid (G i)
    inst✝³ : (i : ι) → Module R (G i)
    f : (i j : ι) → LE.le i j → LinearMap (RingHom.id R) (G i) (G j)
    P : Type u_4
    inst✝² : AddCommMonoid P
    inst✝¹ : Module R P
    g : (i : ι) → LinearMap (RingHom.id R) (G i) P
    Hg : ∀ (i j : ι) (hij : LE.le i j) (x : G i), Eq ((g j) ((f i j hij) x)) ((g i …
    inst✝ : IsDirected ι fun x1 x2 => LE.le x1 x2
    injective : ∀ (i : ι), Function.Injective ⇑(g i)
    h✝ : Nonempty ι
    z w : Module.DirectLimit G f
    eq : Eq ((Module.DirectLimit.lift R ι G f g Hg) z) ((Module.DirectLimit.lift R …
    ⊢ Eq z w
  -/
  obtain ⟨i, x, y, rfl, rfl⟩ := exists_of₂ z w
  /-
    case inr.intro.intro.intro.intro
    R : Type u_1
    inst✝⁷ : Semiring R
    ι : Type u_2
    inst✝⁶ : Preorder ι
    G : ι → Type u_3
    inst✝⁵ : DecidableEq ι
    inst✝⁴ : (i : ι) → AddCommMonoid (G i)
    inst✝³ : (i : ι) → Module R (G i)
    f : (i j : ι) → LE.le i j → LinearMap (RingHom.id R) (G i) (G j)
    P : Type u_4
    inst✝² : AddCommMonoid P
    inst✝¹ : Module R P
    g : (i : ι) → LinearMap (RingHom.id R) (G i) P
    Hg : ∀ (i j : ι) (hij : LE.le i j) (x : G i), Eq ((g j) ((f i j hij) x)) ((g i …
    inst✝ : IsDirected ι fun x1 x2 => LE.le x1 x2
    injective : ∀ (i : ι), Function.Injective ⇑(g i)
    h✝ : Nonempty ι
    i : ι
    x y : G i
    eq : Eq ((Module.DirectLimit.lift R ι G f g Hg) ((Module.DirectLimit.of R ι G  …
    ⊢ Eq ((Module.DirectLimit.of R ι G f i) x) ((Module.DirectLimit.of R ι G f i) y)
  -/
  simp_rw [lift_of] at eq
  /-
    case inr.intro.intro.intro.intro
    R : Type u_1
    inst✝⁷ : Semiring R
    ι : Type u_2
    inst✝⁶ : Preorder ι
    G : ι → Type u_3
    inst✝⁵ : DecidableEq ι
    inst✝⁴ : (i : ι) → AddCommMonoid (G i)
    inst✝³ : (i : ι) → Module R (G i)
    f : (i j : ι) → LE.le i j → LinearMap (RingHom.id R) (G i) (G j)
    P : Type u_4
    inst✝² : AddCommMonoid P
    inst✝¹ : Module R P
    g : (i : ι) → LinearMap (RingHom.id R) (G i) P
    Hg : ∀ (i j : ι) (hij : LE.le i j) (x : G i), Eq ((g j) ((f i j hij) x)) ((g i …
    inst✝ : IsDirected ι fun x1 x2 => LE.le x1 x2
    injective : ∀ (i : ι), Function.Injective ⇑(g i)
    h✝ : Nonempty ι
    i : ι
    x y : G i
    eq : Eq ((g i) x) ((g i) y)
    ⊢ Eq ((Module.DirectLimit.of R ι G f i) x) ((Module.DirectLimit.of R ι G f i) y)
  -/
  rw [injective _ eq]
  /-
    🎉 no goals
  -/


/--
Consider direct limits `lim G` and `lim G'` with direct system `f` and `f'` respectively, any
family of linear maps `gᵢ : Gᵢ ⟶ G'ᵢ` such that `g ∘ f = f' ∘ g` induces a linear map
`lim G ⟶ lim G'`.
-/
def map (g : (i : ι) → G i →ₗ[R] G' i) (hg : ∀ i j h, g j ∘ₗ f i j h = f' i j h ∘ₗ g i) :
    DirectLimit G f →ₗ[R] DirectLimit G' f' :=
  lift _ _ _ _ (fun i ↦ of _ _ _ _ _ ∘ₗ g i) fun i j h g ↦ by
    /-
      R : Type u_1
      inst✝¹⁰ : Semiring R
      ι : Type u_2
      inst✝⁹ : Preorder ι
      G : ι → Type u_3
      inst✝⁸ : DecidableEq ι
      inst✝⁷ : (i : ι) → AddCommMonoid (G i)
      inst✝⁶ : (i : ι) → Module R (G i)
      f : (i j : ι) → LE.le i j → LinearMap (RingHom.id R) (G i) (G j)
      P : Type u_4
      inst✝⁵ : AddCommMonoid P
      inst✝⁴ : Module R P
      g✝¹ : (i : ι) → LinearMap (RingHom.id R) (G i) P
      Hg : ∀ (i j : ι) (hij : LE.le i j) (x : G i), Eq ((g✝¹ j) ((f i j hij) x)) ((g …
      G' : ι → Type u_5
      inst✝³ : (i : ι) → AddCommMonoid (G' i)
      inst✝² : (i : ι) → Module R (G' i)
      f' : (i j : ι) → LE.le i j → LinearMap (RingHom.id R) (G' i) (G' j)
      G'' : ι → Type u_6
      inst✝¹ : (i : ι) → AddCommMonoid (G'' i)
      inst✝ : (i : ι) → Module R (G'' i)
      f'' : (i j : ι) → LE.le i j → LinearMap (RingHom.id R) (G'' i) (G'' j)
      g✝ : (i : ι) → LinearMap (RingHom.id R) (G i) (G' i)
      hg : ∀ (i j : ι) (h : LE.le i j), Eq ((g✝ j).comp (f i j h)) ((f' i j h).comp  …
      i j : ι
      h : LE.le i j
      g : G i
      ⊢ Eq (((fun i => (Module.DirectLimit.of R ι G' f' i).comp (g✝ i)) j) ((f i j h …
    -/
    have eq1 := LinearMap.congr_fun (hg i j h) g
    /-
      R : Type u_1
      inst✝¹⁰ : Semiring R
      ι : Type u_2
      inst✝⁹ : Preorder ι
      G : ι → Type u_3
      inst✝⁸ : DecidableEq ι
      inst✝⁷ : (i : ι) → AddCommMonoid (G i)
      inst✝⁶ : (i : ι) → Module R (G i)
      f : (i j : ι) → LE.le i j → LinearMap (RingHom.id R) (G i) (G j)
      P : Type u_4
      inst✝⁵ : AddCommMonoid P
      inst✝⁴ : Module R P
      g✝¹ : (i : ι) → LinearMap (RingHom.id R) (G i) P
      Hg : ∀ (i j : ι) (hij : LE.le i j) (x : G i), Eq ((g✝¹ j) ((f i j hij) x)) ((g …
      G' : ι → Type u_5
      inst✝³ : (i : ι) → AddCommMonoid (G' i)
      inst✝² : (i : ι) → Module R (G' i)
      f' : (i j : ι) → LE.le i j → LinearMap (RingHom.id R) (G' i) (G' j)
      G'' : ι → Type u_6
      inst✝¹ : (i : ι) → AddCommMonoid (G'' i)
      inst✝ : (i : ι) → Module R (G'' i)
      f'' : (i j : ι) → LE.le i j → LinearMap (RingHom.id R) (G'' i) (G'' j)
      g✝ : (i : ι) → LinearMap (RingHom.id R) (G i) (G' i)
      hg : ∀ (i j : ι) (h : LE.le i j), Eq ((g✝ j).comp (f i j h)) ((f' i j h).comp  …
      i j : ι
      h : LE.le i j
      g : G i
      eq1 : Eq (((g✝ j).comp (f i j h)) g) (((f' i j h).comp (g✝ i)) g)
      ⊢ Eq (((fun i => (Module.DirectLimit.of R ι G' f' i).comp (g✝ i)) j) ((f i j h …
    -/
    simp only [LinearMap.coe_comp, Function.comp_apply] at eq1 ⊢
    /-
      R : Type u_1
      inst✝¹⁰ : Semiring R
      ι : Type u_2
      inst✝⁹ : Preorder ι
      G : ι → Type u_3
      inst✝⁸ : DecidableEq ι
      inst✝⁷ : (i : ι) → AddCommMonoid (G i)
      inst✝⁶ : (i : ι) → Module R (G i)
      f : (i j : ι) → LE.le i j → LinearMap (RingHom.id R) (G i) (G j)
      P : Type u_4
      inst✝⁵ : AddCommMonoid P
      inst✝⁴ : Module R P
      g✝¹ : (i : ι) → LinearMap (RingHom.id R) (G i) P
      Hg : ∀ (i j : ι) (hij : LE.le i j) (x : G i), Eq ((g✝¹ j) ((f i j hij) x)) ((g …
      G' : ι → Type u_5
      inst✝³ : (i : ι) → AddCommMonoid (G' i)
      inst✝² : (i : ι) → Module R (G' i)
      f' : (i j : ι) → LE.le i j → LinearMap (RingHom.id R) (G' i) (G' j)
      G'' : ι → Type u_6
      inst✝¹ : (i : ι) → AddCommMonoid (G'' i)
      inst✝ : (i : ι) → Module R (G'' i)
      f'' : (i j : ι) → LE.le i j → LinearMap (RingHom.id R) (G'' i) (G'' j)
      g✝ : (i : ι) → LinearMap (RingHom.id R) (G i) (G' i)
      hg : ∀ (i j : ι) (h : LE.le i j), Eq ((g✝ j).comp (f i j h)) ((f' i j h).comp  …
      i j : ι
      h : LE.le i j
      g : G i
      eq1 : Eq ((g✝ j) ((f i j h) g)) ((f' i j h) ((g✝ i) g))
      ⊢ Eq ((Module.DirectLimit.of R ι G' f' j) ((g✝ j) ((f i j h) g))) ((Module.Dir …
    -/
    rw [eq1, of_f]
    /-
      🎉 no goals
    -/


@[simp] lemma map_apply_of (g : (i : ι) → G i →ₗ[R] G' i)
    (hg : ∀ i j h, g j ∘ₗ f i j h = f' i j h ∘ₗ g i)
    {i : ι} (x : G i) :
    map g hg (of _ _ G f _ x) = of R ι G' f' i (g i x) :=
  lift_of _ _ _


@[simp] lemma map_id :
    map (fun _ ↦ LinearMap.id) (fun _ _ _ ↦ rfl) = LinearMap.id (R := R) (M := DirectLimit G f) :=
  DFunLike.ext _ _ <| by
    /-
      R : Type u_1
      inst✝⁴ : Semiring R
      ι : Type u_2
      inst✝³ : Preorder ι
      G : ι → Type u_3
      inst✝² : DecidableEq ι
      inst✝¹ : (i : ι) → AddCommMonoid (G i)
      inst✝ : (i : ι) → Module R (G i)
      f : (i j : ι) → LE.le i j → LinearMap (RingHom.id R) (G i) (G j)
      ⊢ ∀ (x : Module.DirectLimit G f), Eq ((Module.DirectLimit.map (fun x => Linear …
    -/
    rintro ⟨x⟩; refine x.induction_on (by simp) (fun _ ↦ map_apply_of _ _) (by simp +contextual)
                /-
                  🎉 no goals
                -/


lemma map_comp (g₁ : (i : ι) → G i →ₗ[R] G' i) (g₂ : (i : ι) → G' i →ₗ[R] G'' i)
    (hg₁ : ∀ i j h, g₁ j ∘ₗ f i j h = f' i j h ∘ₗ g₁ i)
    (hg₂ : ∀ i j h, g₂ j ∘ₗ f' i j h = f'' i j h ∘ₗ g₂ i) :
    (map g₂ hg₂ ∘ₗ map g₁ hg₁ :
      DirectLimit G f →ₗ[R] DirectLimit G'' f'') =
    (map (fun i ↦ g₂ i ∘ₗ g₁ i) fun i j h ↦ by
        /-
          R : Type u_1
          inst✝¹⁰ : Semiring R
          ι : Type u_2
          inst✝⁹ : Preorder ι
          G : ι → Type u_3
          inst✝⁸ : DecidableEq ι
          inst✝⁷ : (i : ι) → AddCommMonoid (G i)
          inst✝⁶ : (i : ι) → Module R (G i)
          f : (i j : ι) → LE.le i j → LinearMap (RingHom.id R) (G i) (G j)
          P : Type u_4
          inst✝⁵ : AddCommMonoid P
          inst✝⁴ : Module R P
          g : (i : ι) → LinearMap (RingHom.id R) (G i) P
          Hg : ∀ (i j : ι) (hij : LE.le i j) (x : G i), Eq ((g j) ((f i j hij) x)) ((g i …
          G' : ι → Type u_5
          inst✝³ : (i : ι) → AddCommMonoid (G' i)
          inst✝² : (i : ι) → Module R (G' i)
          f' : (i j : ι) → LE.le i j → LinearMap (RingHom.id R) (G' i) (G' j)
          G'' : ι → Type u_6
          inst✝¹ : (i : ι) → AddCommMonoid (G'' i)
          inst✝ : (i : ι) → Module R (G'' i)
          f'' : (i j : ι) → LE.le i j → LinearMap (RingHom.id R) (G'' i) (G'' j)
          g₁ : (i : ι) → LinearMap (RingHom.id R) (G i) (G' i)
          g₂ : (i : ι) → LinearMap (RingHom.id R) (G' i) (G'' i)
          hg₁ : ∀ (i j : ι) (h : LE.le i j), Eq ((g₁ j).comp (f i j h)) ((f' i j h).comp …
          hg₂ : ∀ (i j : ι) (h : LE.le i j), Eq ((g₂ j).comp (f' i j h)) ((f'' i j h).co …
          i j : ι
          h : LE.le i j
          ⊢ Eq (((fun i => (g₂ i).comp (g₁ i)) j).comp (f i j h)) ((f'' i j h).comp ((fu …
        -/
        rw [LinearMap.comp_assoc, hg₁ i, ← LinearMap.comp_assoc, hg₂ i, LinearMap.comp_assoc] :
        /-
          🎉 no goals
        -/
      DirectLimit G f →ₗ[R] DirectLimit G'' f'') :=
  DFunLike.ext _ _ <| by
    /-
      R : Type u_1
      inst✝⁸ : Semiring R
      ι : Type u_2
      inst✝⁷ : Preorder ι
      G : ι → Type u_3
      inst✝⁶ : DecidableEq ι
      inst✝⁵ : (i : ι) → AddCommMonoid (G i)
      inst✝⁴ : (i : ι) → Module R (G i)
      f : (i j : ι) → LE.le i j → LinearMap (RingHom.id R) (G i) (G j)
      G' : ι → Type u_5
      inst✝³ : (i : ι) → AddCommMonoid (G' i)
      inst✝² : (i : ι) → Module R (G' i)
      f' : (i j : ι) → LE.le i j → LinearMap (RingHom.id R) (G' i) (G' j)
      G'' : ι → Type u_6
      inst✝¹ : (i : ι) → AddCommMonoid (G'' i)
      inst✝ : (i : ι) → Module R (G'' i)
      f'' : (i j : ι) → LE.le i j → LinearMap (RingHom.id R) (G'' i) (G'' j)
      g₁ : (i : ι) → LinearMap (RingHom.id R) (G i) (G' i)
      g₂ : (i : ι) → LinearMap (RingHom.id R) (G' i) (G'' i)
      hg₁ : ∀ (i j : ι) (h : LE.le i j), Eq ((g₁ j).comp (f i j h)) ((f' i j h).comp …
      hg₂ : ∀ (i j : ι) (h : LE.le i j), Eq ((g₂ j).comp (f' i j h)) ((f'' i j h).co …
      ⊢ ∀ (x : Module.DirectLimit G f), Eq (((Module.DirectLimit.map g₂ hg₂).comp (M …
    -/
    rintro ⟨x⟩; refine x.induction_on (by simp) (fun _ _ ↦ ?_) (by simp +contextual)
    /-
      case mk
      R : Type u_1
      inst✝⁸ : Semiring R
      ι : Type u_2
      inst✝⁷ : Preorder ι
      G : ι → Type u_3
      inst✝⁶ : DecidableEq ι
      inst✝⁵ : (i : ι) → AddCommMonoid (G i)
      inst✝⁴ : (i : ι) → Module R (G i)
      f : (i j : ι) → LE.le i j → LinearMap (RingHom.id R) (G i) (G j)
      G' : ι → Type u_5
      inst✝³ : (i : ι) → AddCommMonoid (G' i)
      inst✝² : (i : ι) → Module R (G' i)
      f' : (i j : ι) → LE.le i j → LinearMap (RingHom.id R) (G' i) (G' j)
      G'' : ι → Type u_6
      inst✝¹ : (i : ι) → AddCommMonoid (G'' i)
      inst✝ : (i : ι) → Module R (G'' i)
      f'' : (i j : ι) → LE.le i j → LinearMap (RingHom.id R) (G'' i) (G'' j)
      g₁ : (i : ι) → LinearMap (RingHom.id R) (G i) (G' i)
      g₂ : (i : ι) → LinearMap (RingHom.id R) (G' i) (G'' i)
      hg₁ : ∀ (i j : ι) (h : LE.le i j), Eq ((g₁ j).comp (f i j h)) ((f' i j h).comp …
      hg₂ : ∀ (i j : ι) (h : LE.le i j), Eq ((g₂ j).comp (f' i j h)) ((f'' i j h).co …
      x✝² : Module.DirectLimit G f
      x : DirectSum ι G
      x✝¹ : ι
      x✝ : G x✝¹
      ⊢ Eq (((Module.DirectLimit.map g₂ hg₂).comp (Module.DirectLimit.map g₁ hg₁)) ( …
    -/
    show map g₂ hg₂ (map g₁ hg₁ <| of _ _ _ _ _ _) = map _ _ (of _ _ _ _ _ _)
    /-
      case mk
      R : Type u_1
      inst✝⁸ : Semiring R
      ι : Type u_2
      inst✝⁷ : Preorder ι
      G : ι → Type u_3
      inst✝⁶ : DecidableEq ι
      inst✝⁵ : (i : ι) → AddCommMonoid (G i)
      inst✝⁴ : (i : ι) → Module R (G i)
      f : (i j : ι) → LE.le i j → LinearMap (RingHom.id R) (G i) (G j)
      G' : ι → Type u_5
      inst✝³ : (i : ι) → AddCommMonoid (G' i)
      inst✝² : (i : ι) → Module R (G' i)
      f' : (i j : ι) → LE.le i j → LinearMap (RingHom.id R) (G' i) (G' j)
      G'' : ι → Type u_6
      inst✝¹ : (i : ι) → AddCommMonoid (G'' i)
      inst✝ : (i : ι) → Module R (G'' i)
      f'' : (i j : ι) → LE.le i j → LinearMap (RingHom.id R) (G'' i) (G'' j)
      g₁ : (i : ι) → LinearMap (RingHom.id R) (G i) (G' i)
      g₂ : (i : ι) → LinearMap (RingHom.id R) (G' i) (G'' i)
      hg₁ : ∀ (i j : ι) (h : LE.le i j), Eq ((g₁ j).comp (f i j h)) ((f' i j h).comp …
      hg₂ : ∀ (i j : ι) (h : LE.le i j), Eq ((g₂ j).comp (f' i j h)) ((f'' i j h).co …
      x✝² : Module.DirectLimit G f
      x : DirectSum ι G
      x✝¹ : ι
      x✝ : G x✝¹
      ⊢ Eq ((Module.DirectLimit.map g₂ hg₂) ((Module.DirectLimit.map g₁ hg₁) ((Modul …
    -/
    simp_rw [map_apply_of]; rfl
                            /-
                              🎉 no goals
                            -/


open LinearEquiv LinearMap in
/--
Consider direct limits `lim G` and `lim G'` with direct system `f` and `f'` respectively, any
family of equivalences `eᵢ : Gᵢ ≅ G'ᵢ` such that `e ∘ f = f' ∘ e` induces an equivalence
`lim G ≅ lim G'`.
-/
def congr (e : (i : ι) → G i ≃ₗ[R] G' i) (he : ∀ i j h, e j ∘ₗ f i j h = f' i j h ∘ₗ e i) :
    DirectLimit G f ≃ₗ[R] DirectLimit G' f' :=
  LinearEquiv.ofLinear (map (e ·) he)
    (map (fun i ↦ (e i).symm) fun i j h ↦ by
      rw [toLinearMap_symm_comp_eq, ← comp_assoc, he i, comp_assoc, comp_coe, symm_trans_self,
        refl_toLinearMap, comp_id])
        /-
          R : Type u_1
          inst✝¹⁰ : Semiring R
          ι : Type u_2
          inst✝⁹ : Preorder ι
          G : ι → Type u_3
          inst✝⁸ : DecidableEq ι
          inst✝⁷ : (i : ι) → AddCommMonoid (G i)
          inst✝⁶ : (i : ι) → Module R (G i)
          f : (i j : ι) → LE.le i j → LinearMap (RingHom.id R) (G i) (G j)
          P : Type u_4
          inst✝⁵ : AddCommMonoid P
          inst✝⁴ : Module R P
          g : (i : ι) → LinearMap (RingHom.id R) (G i) P
          Hg : ∀ (i j : ι) (hij : LE.le i j) (x : G i), Eq ((g j) ((f i j hij) x)) ((g i …
          G' : ι → Type u_5
          inst✝³ : (i : ι) → AddCommMonoid (G' i)
          inst✝² : (i : ι) → Module R (G' i)
          f' : (i j : ι) → LE.le i j → LinearMap (RingHom.id R) (G' i) (G' j)
          G'' : ι → Type u_6
          inst✝¹ : (i : ι) → AddCommMonoid (G'' i)
          inst✝ : (i : ι) → Module R (G'' i)
          f'' : (i j : ι) → LE.le i j → LinearMap (RingHom.id R) (G'' i) (G'' j)
          e : (i : ι) → LinearEquiv (RingHom.id R) (G i) (G' i)
          he : ∀ (i j : ι) (h : LE.le i j), Eq ((↑(e j)).comp (f i j h)) ((f' i j h).com …
          ⊢ Eq ((Module.DirectLimit.map (fun x => ↑(e x)) he).comp (Module.DirectLimit.m …
        -/
        /-
          🎉 no goals
        -/
    (by simp [map_comp]) (by simp [map_comp])
                             /-
                               🎉 no goals
                             -/


lemma congr_apply_of (e : (i : ι) → G i ≃ₗ[R] G' i) (he : ∀ i j h, e j ∘ₗ f i j h = f' i j h ∘ₗ e i)
    {i : ι} (g : G i) :
    congr e he (of _ _ G f i g) = of _ _ G' f' i (e i g) :=
  map_apply_of _ he _


open LinearEquiv LinearMap in
lemma congr_symm_apply_of (e : (i : ι) → G i ≃ₗ[R] G' i)
    (he : ∀ i j h, e j ∘ₗ f i j h = f' i j h ∘ₗ e i) {i : ι} (g : G' i) :
    (congr e he).symm (of _ _ G' f' i g) = of _ _ G f i ((e i).symm g) :=
  map_apply_of _ (fun i j h ↦ by
    rw [toLinearMap_symm_comp_eq, ← comp_assoc, he i, comp_assoc, comp_coe, symm_trans_self,
      refl_toLinearMap, comp_id]) _


/-- The direct limit constructed as a quotient of the direct sum is isomorphic to
the direct limit constructed as a quotient of the disjoint union. -/
def linearEquiv : DirectLimit G f ≃ₗ[R] _root_.DirectLimit G f :=
  .ofLinear (lift _ _ _ _ (Module.of _ _ _ _) fun _ _ _ _ ↦ .symm <| eq_of_le ..)
    (Module.lift _ _ _ _ (of _ _ _ _) fun _ _ _ _ ↦ of_f ..)
        /-
          R : Type u_1
          inst✝⁷ : Semiring R
          ι : Type u_2
          inst✝⁶ : Preorder ι
          G : ι → Type u_3
          inst✝⁵ : DecidableEq ι
          inst✝⁴ : (i : ι) → AddCommMonoid (G i)
          inst✝³ : (i : ι) → Module R (G i)
          f : (i j : ι) → LE.le i j → LinearMap (RingHom.id R) (G i) (G j)
          inst✝² : Nonempty ι
          inst✝¹ : IsDirected ι fun x1 x2 => LE.le x1 x2
          inst✝ : DirectedSystem G fun x1 x2 x3 => ⇑(f x1 x2 x3)
          ⊢ Eq ((Module.DirectLimit.lift R ι G f (DirectLimit.Module.of R ι G f) ⋯).comp …
        -/
    (by ext ⟨_⟩; rw [← Quotient.mk]; simp [Module.lift, _root_.DirectLimit.lift_def]; rfl) <| by
                                                                                      /-
                                                                                        🎉 no goals
                                                                                      -/
      /-
        R : Type u_1
        inst✝⁷ : Semiring R
        ι : Type u_2
        inst✝⁶ : Preorder ι
        G : ι → Type u_3
        inst✝⁵ : DecidableEq ι
        inst✝⁴ : (i : ι) → AddCommMonoid (G i)
        inst✝³ : (i : ι) → Module R (G i)
        f : (i j : ι) → LE.le i j → LinearMap (RingHom.id R) (G i) (G j)
        inst✝² : Nonempty ι
        inst✝¹ : IsDirected ι fun x1 x2 => LE.le x1 x2
        inst✝ : DirectedSystem G fun x1 x2 x3 => ⇑(f x1 x2 x3)
        ⊢ Eq ((DirectLimit.Module.lift R ι G f (Module.DirectLimit.of R ι G f) ⋯).comp …
      -/
      ext ⟨x⟩; refine x.induction_on (by simp) (fun i x ↦ ?_) (by simp+contextual)
      /-
        case h.mk
        R : Type u_1
        inst✝⁷ : Semiring R
        ι : Type u_2
        inst✝⁶ : Preorder ι
        G : ι → Type u_3
        inst✝⁵ : DecidableEq ι
        inst✝⁴ : (i : ι) → AddCommMonoid (G i)
        inst✝³ : (i : ι) → Module R (G i)
        f : (i j : ι) → LE.le i j → LinearMap (RingHom.id R) (G i) (G j)
        inst✝² : Nonempty ι
        inst✝¹ : IsDirected ι fun x1 x2 => LE.le x1 x2
        inst✝ : DirectedSystem G fun x1 x2 x3 => ⇑(f x1 x2 x3)
        x✝¹ : Module.DirectLimit G f
        x✝ : DirectSum ι G
        i : ι
        x : G i
        ⊢ Eq (((DirectLimit.Module.lift R ι G f (Module.DirectLimit.of R ι G f) ⋯).com …
      -/
      rw [quotMk_of, LinearMap.comp_apply, lift_of, Module.lift_of, LinearMap.id_apply]
      /-
        🎉 no goals
      -/


theorem linearEquiv_of {i g} : linearEquiv _ _ (of _ _ G f i g) = ⟦⟨i, g⟩⟧ := by
  /-
    R : Type u_1
    inst✝⁷ : Semiring R
    ι : Type u_2
    inst✝⁶ : Preorder ι
    G : ι → Type u_3
    inst✝⁵ : DecidableEq ι
    inst✝⁴ : (i : ι) → AddCommMonoid (G i)
    inst✝³ : (i : ι) → Module R (G i)
    f : (i j : ι) → LE.le i j → LinearMap (RingHom.id R) (G i) (G j)
    inst✝² : Nonempty ι
    inst✝¹ : IsDirected ι fun x1 x2 => LE.le x1 x2
    inst✝ : DirectedSystem G fun x1 x2 x3 => ⇑(f x1 x2 x3)
    i : ι
    g : G i
    ⊢ Eq ((Module.DirectLimit.linearEquiv G f) ((Module.DirectLimit.of R ι G f i)  …
  -/
  simp [linearEquiv]; rfl
                      /-
                        🎉 no goals
                      -/


theorem linearEquiv_symm_mk {g} : (linearEquiv _ _).symm ⟦g⟧ = of _ _ G f g.1 g.2 := rfl


theorem exists_eq_of_of_eq {i x y} (h : of R ι G f i x = of R ι G f i y) :
    ∃ j hij, f i j hij x = f i j hij y := by
  /-
    R : Type u_1
    inst✝⁶ : Semiring R
    ι : Type u_2
    inst✝⁵ : Preorder ι
    G : ι → Type u_3
    inst✝⁴ : DecidableEq ι
    inst✝³ : (i : ι) → AddCommMonoid (G i)
    inst✝² : (i : ι) → Module R (G i)
    f : (i j : ι) → LE.le i j → LinearMap (RingHom.id R) (G i) (G j)
    inst✝¹ : DirectedSystem G fun x1 x2 x3 => ⇑(f x1 x2 x3)
    inst✝ : IsDirected ι fun x1 x2 => LE.le x1 x2
    i : ι
    x y : G i
    h : Eq ((Module.DirectLimit.of R ι G f i) x) ((Module.DirectLimit.of R ι G f i …
    ⊢ Exists fun j => Exists fun hij => Eq ((f i j hij) x) ((f i j hij) y)
  -/
  have := Nonempty.intro i
  /-
    R : Type u_1
    inst✝⁶ : Semiring R
    ι : Type u_2
    inst✝⁵ : Preorder ι
    G : ι → Type u_3
    inst✝⁴ : DecidableEq ι
    inst✝³ : (i : ι) → AddCommMonoid (G i)
    inst✝² : (i : ι) → Module R (G i)
    f : (i j : ι) → LE.le i j → LinearMap (RingHom.id R) (G i) (G j)
    inst✝¹ : DirectedSystem G fun x1 x2 x3 => ⇑(f x1 x2 x3)
    inst✝ : IsDirected ι fun x1 x2 => LE.le x1 x2
    i : ι
    x y : G i
    h : Eq ((Module.DirectLimit.of R ι G f i) x) ((Module.DirectLimit.of R ι G f i …
    this : Nonempty ι
    ⊢ Exists fun j => Exists fun hij => Eq ((f i j hij) x) ((f i j hij) y)
  -/
  apply_fun linearEquiv _ _ at h
  /-
    R : Type u_1
    inst✝⁶ : Semiring R
    ι : Type u_2
    inst✝⁵ : Preorder ι
    G : ι → Type u_3
    inst✝⁴ : DecidableEq ι
    inst✝³ : (i : ι) → AddCommMonoid (G i)
    inst✝² : (i : ι) → Module R (G i)
    f : (i j : ι) → LE.le i j → LinearMap (RingHom.id R) (G i) (G j)
    inst✝¹ : DirectedSystem G fun x1 x2 x3 => ⇑(f x1 x2 x3)
    inst✝ : IsDirected ι fun x1 x2 => LE.le x1 x2
    i : ι
    x y : G i
    this : Nonempty ι
    h : Eq ((Module.DirectLimit.linearEquiv G f) ((Module.DirectLimit.of R ι G f i …
    ⊢ Exists fun j => Exists fun hij => Eq ((f i j hij) x) ((f i j hij) y)
  -/
  simp_rw [linearEquiv_of] at h
  /-
    R : Type u_1
    inst✝⁶ : Semiring R
    ι : Type u_2
    inst✝⁵ : Preorder ι
    G : ι → Type u_3
    inst✝⁴ : DecidableEq ι
    inst✝³ : (i : ι) → AddCommMonoid (G i)
    inst✝² : (i : ι) → Module R (G i)
    f : (i j : ι) → LE.le i j → LinearMap (RingHom.id R) (G i) (G j)
    inst✝¹ : DirectedSystem G fun x1 x2 x3 => ⇑(f x1 x2 x3)
    inst✝ : IsDirected ι fun x1 x2 => LE.le x1 x2
    i : ι
    x y : G i
    this : Nonempty ι
    h : Eq (Quotient.mk (DirectLimit.setoid f) ⟨i, x⟩) (Quotient.mk (DirectLimit.s …
    ⊢ Exists fun j => Exists fun hij => Eq ((f i j hij) x) ((f i j hij) y)
  -/
  have ⟨j, h⟩ := Quotient.exact h
  /-
    R : Type u_1
    inst✝⁶ : Semiring R
    ι : Type u_2
    inst✝⁵ : Preorder ι
    G : ι → Type u_3
    inst✝⁴ : DecidableEq ι
    inst✝³ : (i : ι) → AddCommMonoid (G i)
    inst✝² : (i : ι) → Module R (G i)
    f : (i j : ι) → LE.le i j → LinearMap (RingHom.id R) (G i) (G j)
    inst✝¹ : DirectedSystem G fun x1 x2 x3 => ⇑(f x1 x2 x3)
    inst✝ : IsDirected ι fun x1 x2 => LE.le x1 x2
    i : ι
    x y : G i
    this : Nonempty ι
    h✝ : Eq (Quotient.mk (DirectLimit.setoid f) ⟨i, x⟩) (Quotient.mk (DirectLimit. …
    j : ι
    h : Exists fun hx => Exists fun hy => Eq ((f ⟨i, x⟩.fst j hx) ⟨i, x⟩.snd) ((f  …
    ⊢ Exists fun j => Exists fun hij => Eq ((f i j hij) x) ((f i j hij) y)
  -/
  exact ⟨j, h.1, h.2.2⟩
  /-
    🎉 no goals
  -/


/-- A component that corresponds to zero in the direct limit is already zero in some
bigger module in the directed system. -/
theorem of.zero_exact {i x} (H : of R ι G f i x = 0) :
    ∃ j hij, f i j hij x = (0 : G j) := by
  /-
    R : Type u_1
    inst✝⁶ : Semiring R
    ι : Type u_2
    inst✝⁵ : Preorder ι
    G : ι → Type u_3
    inst✝⁴ : DecidableEq ι
    inst✝³ : (i : ι) → AddCommMonoid (G i)
    inst✝² : (i : ι) → Module R (G i)
    f : (i j : ι) → LE.le i j → LinearMap (RingHom.id R) (G i) (G j)
    inst✝¹ : DirectedSystem G fun x1 x2 x3 => ⇑(f x1 x2 x3)
    inst✝ : IsDirected ι fun x1 x2 => LE.le x1 x2
    i : ι
    x : G i
    H : Eq ((Module.DirectLimit.of R ι G f i) x) 0
    ⊢ Exists fun j => Exists fun hij => Eq ((f i j hij) x) 0
  -/
  convert exists_eq_of_of_eq (H.trans (map_zero <| _).symm)
  /-
    case h.e'_2.h.h.e'_2.h.h.e'_3
    R : Type u_1
    inst✝⁶ : Semiring R
    ι : Type u_2
    inst✝⁵ : Preorder ι
    G : ι → Type u_3
    inst✝⁴ : DecidableEq ι
    inst✝³ : (i : ι) → AddCommMonoid (G i)
    inst✝² : (i : ι) → Module R (G i)
    f : (i j : ι) → LE.le i j → LinearMap (RingHom.id R) (G i) (G j)
    inst✝¹ : DirectedSystem G fun x1 x2 x3 => ⇑(f x1 x2 x3)
    inst✝ : IsDirected ι fun x1 x2 => LE.le x1 x2
    i : ι
    x : G i
    H : Eq ((Module.DirectLimit.of R ι G f i) x) 0
    x✝¹ : ι
    x✝ : LE.le i x✝¹
    ⊢ Eq 0 ((f i x✝¹ x✝) 0)
  -/
  rw [map_zero]
  /-
    🎉 no goals
  -/


/-- The direct limit of a directed system is the abelian groups glued together along the maps. -/
def DirectLimit [DecidableEq ι] (f : ∀ i j, i ≤ j → G i →+ G j) : Type _ :=
  @Module.DirectLimit ℕ _ ι _ G _ _ (fun i j hij ↦ (f i j hij).toNatLinearMap) _


local instance directedSystem [h : DirectedSystem G fun i j h ↦ f i j h] :
    DirectedSystem G fun i j hij ↦ (f i j hij).toNatLinearMap :=
  h


instance : AddCommMonoid (DirectLimit G f) :=
  Module.DirectLimit.addCommMonoid G fun i j hij ↦ (f i j hij).toNatLinearMap


instance addCommGroup (G : ι → Type*) [∀ i, AddCommGroup (G i)]
    (f : ∀ i j, i ≤ j → G i →+ G j) : AddCommGroup (DirectLimit G f) :=
  Module.DirectLimit.addCommGroup G fun i j hij ↦ (f i j hij).toNatLinearMap


instance : Inhabited (DirectLimit G f) :=
  ⟨0⟩


instance [IsEmpty ι] : Unique (DirectLimit G f) := Module.DirectLimit.unique _ _


/-- The canonical map from a component to the direct limit. -/
def of (i) : G i →+ DirectLimit G f :=
  (Module.DirectLimit.of ℕ ι G _ i).toAddMonoidHom


@[simp]
theorem of_f {i j} (hij) (x) : of G f j (f i j hij x) = of G f i x :=
  Module.DirectLimit.of_f


@[elab_as_elim]
protected theorem induction_on [Nonempty ι] [IsDirected ι (· ≤ ·)] {C : DirectLimit G f → Prop}
    (z : DirectLimit G f) (ih : ∀ i x, C (of G f i x)) : C z :=
  Module.DirectLimit.induction_on z ih


/-- A component that corresponds to zero in the direct limit is already zero in some
bigger module in the directed system. -/
theorem of.zero_exact [IsDirected ι (· ≤ ·)] [DirectedSystem G fun i j h ↦ f i j h] (i x)
    (h : of G f i x = 0) : ∃ j hij, f i j hij x = 0 :=
  Module.DirectLimit.of.zero_exact h


/-- The universal property of the direct limit: maps from the components to another abelian group
that respect the directed system structure (i.e. make some diagram commute) give rise
to a unique map out of the direct limit. -/
def lift : DirectLimit G f →+ P :=
  (Module.DirectLimit.lift ℕ ι G (fun i j hij ↦ (f i j hij).toNatLinearMap)
    (fun i ↦ (g i).toNatLinearMap) Hg).toAddMonoidHom


@[simp]
theorem lift_of (i x) : lift G f P g Hg (of G f i x) = g i x :=
  Module.DirectLimit.lift_of
    -- Note: had to make these arguments explicit https://github.com/leanprover-community/mathlib4/pull/8386
    (f := fun i j hij ↦ (f i j hij).toNatLinearMap)
    (fun i ↦ (g i).toNatLinearMap)
    Hg
    x


theorem lift_unique (F : DirectLimit G f →+ P) (x) :
                                                                     /-
                                                                       R : Type u_1
                                                                       inst✝⁴ : Semiring R
                                                                       ι : Type u_2
                                                                       inst✝³ : Preorder ι
                                                                       G : ι → Type u_3
                                                                       inst✝² : (i : ι) → AddCommMonoid (G i)
                                                                       f : (i j : ι) → LE.le i j → AddMonoidHom (G i) (G j)
                                                                       inst✝¹ : DecidableEq ι
                                                                       P : Type u_4
                                                                       inst✝ : AddCommMonoid P
                                                                       g : (i : ι) → AddMonoidHom (G i) P
                                                                       Hg : ∀ (i j : ι) (hij : LE.le i j) (x : G i), Eq ((g j) ((f i j hij) x)) ((g i …
                                                                       F : AddMonoidHom (AddCommGroup.DirectLimit G f) P
                                                                       x✝ : AddCommGroup.DirectLimit G f
                                                                       i j : ι
                                                                       hij : LE.le i j
                                                                       x : G i
                                                                       ⊢ Eq (((fun i => F.comp (AddCommGroup.DirectLimit.of G f i)) j) ((f i j hij) x …
                                                                     -/
    F x = lift G f P (fun i ↦ F.comp (of G f i)) (fun i j hij x ↦ by simp) x := by
                                                                     /-
                                                                       🎉 no goals
                                                                     -/
  /-
    ι : Type u_2
    inst✝³ : Preorder ι
    G : ι → Type u_3
    inst✝² : (i : ι) → AddCommMonoid (G i)
    f : (i j : ι) → LE.le i j → AddMonoidHom (G i) (G j)
    inst✝¹ : DecidableEq ι
    P : Type u_4
    inst✝ : AddCommMonoid P
    F : AddMonoidHom (AddCommGroup.DirectLimit G f) P
    x : AddCommGroup.DirectLimit G f
    ⊢ Eq (F x) ((AddCommGroup.DirectLimit.lift G f P (fun i => F.comp (AddCommGrou …
  -/
  rcases x with ⟨x⟩
  /-
    case mk
    ι : Type u_2
    inst✝³ : Preorder ι
    G : ι → Type u_3
    inst✝² : (i : ι) → AddCommMonoid (G i)
    f : (i j : ι) → LE.le i j → AddMonoidHom (G i) (G j)
    inst✝¹ : DecidableEq ι
    P : Type u_4
    inst✝ : AddCommMonoid P
    F : AddMonoidHom (AddCommGroup.DirectLimit G f) P
    x✝ : AddCommGroup.DirectLimit G f
    x : DirectSum ι G
    ⊢ Eq (F (Quot.mk (⇑(addConGen (Module.DirectLimit.Eqv fun i j hij => (f i j hi …
  -/
  exact x.induction_on (by simp) (fun _ _ ↦ .symm <| lift_of ..) (by simp +contextual)
  /-
    🎉 no goals
  -/


lemma lift_injective [IsDirected ι (· ≤ ·)]
    (injective : ∀ i, Function.Injective <| g i) :
    Function.Injective (lift G f P g Hg) :=
  Module.DirectLimit.lift_injective (f := fun i j hij ↦ (f i j hij).toNatLinearMap) _ Hg injective


/--
Consider direct limits `lim G` and `lim G'` with direct system `f` and `f'` respectively, any
family of group homomorphisms `gᵢ : Gᵢ ⟶ G'ᵢ` such that `g ∘ f = f' ∘ g` induces a group
homomorphism `lim G ⟶ lim G'`.
-/
def map (g : (i : ι) → G i →+ G' i)
    (hg : ∀ i j h, (g j).comp (f i j h) = (f' i j h).comp (g i)) :
    DirectLimit G f →+ DirectLimit G' f' :=
  lift _ _ _ (fun i ↦ (of _ _ _).comp (g i)) fun i j h g ↦ by
    /-
      R : Type u_1
      inst✝⁶ : Semiring R
      ι : Type u_2
      inst✝⁵ : Preorder ι
      G : ι → Type u_3
      inst✝⁴ : (i : ι) → AddCommMonoid (G i)
      f : (i j : ι) → LE.le i j → AddMonoidHom (G i) (G j)
      inst✝³ : DecidableEq ι
      P : Type u_4
      inst✝² : AddCommMonoid P
      g✝¹ : (i : ι) → AddMonoidHom (G i) P
      Hg : ∀ (i j : ι) (hij : LE.le i j) (x : G i), Eq ((g✝¹ j) ((f i j hij) x)) ((g …
      G' : ι → Type u_5
      inst✝¹ : (i : ι) → AddCommMonoid (G' i)
      f' : (i j : ι) → LE.le i j → AddMonoidHom (G' i) (G' j)
      G'' : ι → Type u_6
      inst✝ : (i : ι) → AddCommMonoid (G'' i)
      f'' : (i j : ι) → LE.le i j → AddMonoidHom (G'' i) (G'' j)
      g✝ : (i : ι) → AddMonoidHom (G i) (G' i)
      hg : ∀ (i j : ι) (h : LE.le i j), Eq ((g✝ j).comp (f i j h)) ((f' i j h).comp  …
      i j : ι
      h : LE.le i j
      g : G i
      ⊢ Eq (((fun i => (AddCommGroup.DirectLimit.of G' f' i).comp (g✝ i)) j) ((f i j …
    -/
    have eq1 := DFunLike.congr_fun (hg i j h) g
    /-
      R : Type u_1
      inst✝⁶ : Semiring R
      ι : Type u_2
      inst✝⁵ : Preorder ι
      G : ι → Type u_3
      inst✝⁴ : (i : ι) → AddCommMonoid (G i)
      f : (i j : ι) → LE.le i j → AddMonoidHom (G i) (G j)
      inst✝³ : DecidableEq ι
      P : Type u_4
      inst✝² : AddCommMonoid P
      g✝¹ : (i : ι) → AddMonoidHom (G i) P
      Hg : ∀ (i j : ι) (hij : LE.le i j) (x : G i), Eq ((g✝¹ j) ((f i j hij) x)) ((g …
      G' : ι → Type u_5
      inst✝¹ : (i : ι) → AddCommMonoid (G' i)
      f' : (i j : ι) → LE.le i j → AddMonoidHom (G' i) (G' j)
      G'' : ι → Type u_6
      inst✝ : (i : ι) → AddCommMonoid (G'' i)
      f'' : (i j : ι) → LE.le i j → AddMonoidHom (G'' i) (G'' j)
      g✝ : (i : ι) → AddMonoidHom (G i) (G' i)
      hg : ∀ (i j : ι) (h : LE.le i j), Eq ((g✝ j).comp (f i j h)) ((f' i j h).comp  …
      i j : ι
      h : LE.le i j
      g : G i
      eq1 : Eq (((g✝ j).comp (f i j h)) g) (((f' i j h).comp (g✝ i)) g)
      ⊢ Eq (((fun i => (AddCommGroup.DirectLimit.of G' f' i).comp (g✝ i)) j) ((f i j …
    -/
    simp only [AddMonoidHom.coe_comp, Function.comp_apply] at eq1 ⊢
    /-
      R : Type u_1
      inst✝⁶ : Semiring R
      ι : Type u_2
      inst✝⁵ : Preorder ι
      G : ι → Type u_3
      inst✝⁴ : (i : ι) → AddCommMonoid (G i)
      f : (i j : ι) → LE.le i j → AddMonoidHom (G i) (G j)
      inst✝³ : DecidableEq ι
      P : Type u_4
      inst✝² : AddCommMonoid P
      g✝¹ : (i : ι) → AddMonoidHom (G i) P
      Hg : ∀ (i j : ι) (hij : LE.le i j) (x : G i), Eq ((g✝¹ j) ((f i j hij) x)) ((g …
      G' : ι → Type u_5
      inst✝¹ : (i : ι) → AddCommMonoid (G' i)
      f' : (i j : ι) → LE.le i j → AddMonoidHom (G' i) (G' j)
      G'' : ι → Type u_6
      inst✝ : (i : ι) → AddCommMonoid (G'' i)
      f'' : (i j : ι) → LE.le i j → AddMonoidHom (G'' i) (G'' j)
      g✝ : (i : ι) → AddMonoidHom (G i) (G' i)
      hg : ∀ (i j : ι) (h : LE.le i j), Eq ((g✝ j).comp (f i j h)) ((f' i j h).comp  …
      i j : ι
      h : LE.le i j
      g : G i
      eq1 : Eq ((g✝ j) ((f i j h) g)) ((f' i j h) ((g✝ i) g))
      ⊢ Eq ((AddCommGroup.DirectLimit.of G' f' j) ((g✝ j) ((f i j h) g))) ((AddCommG …
    -/
    rw [eq1, of_f]
    /-
      🎉 no goals
    -/


@[simp] lemma map_apply_of (g : (i : ι) → G i →+ G' i)
    (hg : ∀ i j h, (g j).comp (f i j h) = (f' i j h).comp (g i))
    {i : ι} (x : G i) :
    map g hg (of G f _ x) = of G' f' i (g i x) :=
  lift_of _ _ _ _ _


@[simp] lemma map_id :
    map (fun _ ↦ AddMonoidHom.id _) (fun _ _ _ ↦ rfl) = AddMonoidHom.id (DirectLimit G f) :=
  DFunLike.ext _ _ <| by
    /-
      ι : Type u_2
      inst✝² : Preorder ι
      G : ι → Type u_3
      inst✝¹ : (i : ι) → AddCommMonoid (G i)
      f : (i j : ι) → LE.le i j → AddMonoidHom (G i) (G j)
      inst✝ : DecidableEq ι
      ⊢ ∀ (x : AddCommGroup.DirectLimit G f), Eq ((AddCommGroup.DirectLimit.map (fun …
    -/
    rintro ⟨x⟩; refine x.induction_on (by simp) (fun _ ↦ map_apply_of _ _) (by simp +contextual)
                /-
                  🎉 no goals
                -/


lemma map_comp (g₁ : (i : ι) → G i →+ G' i) (g₂ : (i : ι) → G' i →+ G'' i)
    (hg₁ : ∀ i j h, (g₁ j).comp (f i j h) = (f' i j h).comp (g₁ i))
    (hg₂ : ∀ i j h, (g₂ j).comp (f' i j h) = (f'' i j h).comp (g₂ i)) :
    ((map g₂ hg₂).comp (map g₁ hg₁) :
      DirectLimit G f →+ DirectLimit G'' f'') =
    (map (fun i ↦ (g₂ i).comp (g₁ i)) fun i j h ↦ by
      rw [AddMonoidHom.comp_assoc, hg₁ i, ← AddMonoidHom.comp_assoc, hg₂ i,
        AddMonoidHom.comp_assoc] :
      DirectLimit G f →+ DirectLimit G'' f'') :=
  DFunLike.ext _ _ <| by
    /-
      ι : Type u_2
      inst✝⁴ : Preorder ι
      G : ι → Type u_3
      inst✝³ : (i : ι) → AddCommMonoid (G i)
      f : (i j : ι) → LE.le i j → AddMonoidHom (G i) (G j)
      inst✝² : DecidableEq ι
      G' : ι → Type u_5
      inst✝¹ : (i : ι) → AddCommMonoid (G' i)
      f' : (i j : ι) → LE.le i j → AddMonoidHom (G' i) (G' j)
      G'' : ι → Type u_6
      inst✝ : (i : ι) → AddCommMonoid (G'' i)
      f'' : (i j : ι) → LE.le i j → AddMonoidHom (G'' i) (G'' j)
      g₁ : (i : ι) → AddMonoidHom (G i) (G' i)
      g₂ : (i : ι) → AddMonoidHom (G' i) (G'' i)
      hg₁ : ∀ (i j : ι) (h : LE.le i j), Eq ((g₁ j).comp (f i j h)) ((f' i j h).comp …
      hg₂ : ∀ (i j : ι) (h : LE.le i j), Eq ((g₂ j).comp (f' i j h)) ((f'' i j h).co …
      ⊢ ∀ (x : AddCommGroup.DirectLimit G f), Eq (((AddCommGroup.DirectLimit.map g₂  …
    -/
    rintro ⟨x⟩; refine x.induction_on (by simp) (fun _ _ ↦ ?_) (by simp +contextual)
    /-
      case mk
      ι : Type u_2
      inst✝⁴ : Preorder ι
      G : ι → Type u_3
      inst✝³ : (i : ι) → AddCommMonoid (G i)
      f : (i j : ι) → LE.le i j → AddMonoidHom (G i) (G j)
      inst✝² : DecidableEq ι
      G' : ι → Type u_5
      inst✝¹ : (i : ι) → AddCommMonoid (G' i)
      f' : (i j : ι) → LE.le i j → AddMonoidHom (G' i) (G' j)
      G'' : ι → Type u_6
      inst✝ : (i : ι) → AddCommMonoid (G'' i)
      f'' : (i j : ι) → LE.le i j → AddMonoidHom (G'' i) (G'' j)
      g₁ : (i : ι) → AddMonoidHom (G i) (G' i)
      g₂ : (i : ι) → AddMonoidHom (G' i) (G'' i)
      hg₁ : ∀ (i j : ι) (h : LE.le i j), Eq ((g₁ j).comp (f i j h)) ((f' i j h).comp …
      hg₂ : ∀ (i j : ι) (h : LE.le i j), Eq ((g₂ j).comp (f' i j h)) ((f'' i j h).co …
      x✝² : AddCommGroup.DirectLimit G f
      x : DirectSum ι G
      x✝¹ : ι
      x✝ : G x✝¹
      ⊢ Eq (((AddCommGroup.DirectLimit.map g₂ hg₂).comp (AddCommGroup.DirectLimit.ma …
    -/
    show map g₂ hg₂ (map g₁ hg₁ <| of _ _ _ _) = map _ _ (of _ _ _ _)
    /-
      case mk
      ι : Type u_2
      inst✝⁴ : Preorder ι
      G : ι → Type u_3
      inst✝³ : (i : ι) → AddCommMonoid (G i)
      f : (i j : ι) → LE.le i j → AddMonoidHom (G i) (G j)
      inst✝² : DecidableEq ι
      G' : ι → Type u_5
      inst✝¹ : (i : ι) → AddCommMonoid (G' i)
      f' : (i j : ι) → LE.le i j → AddMonoidHom (G' i) (G' j)
      G'' : ι → Type u_6
      inst✝ : (i : ι) → AddCommMonoid (G'' i)
      f'' : (i j : ι) → LE.le i j → AddMonoidHom (G'' i) (G'' j)
      g₁ : (i : ι) → AddMonoidHom (G i) (G' i)
      g₂ : (i : ι) → AddMonoidHom (G' i) (G'' i)
      hg₁ : ∀ (i j : ι) (h : LE.le i j), Eq ((g₁ j).comp (f i j h)) ((f' i j h).comp …
      hg₂ : ∀ (i j : ι) (h : LE.le i j), Eq ((g₂ j).comp (f' i j h)) ((f'' i j h).co …
      x✝² : AddCommGroup.DirectLimit G f
      x : DirectSum ι G
      x✝¹ : ι
      x✝ : G x✝¹
      ⊢ Eq ((AddCommGroup.DirectLimit.map g₂ hg₂) ((AddCommGroup.DirectLimit.map g₁  …
    -/
    simp_rw [map_apply_of]; rfl
                            /-
                              🎉 no goals
                            -/


/--
Consider direct limits `lim G` and `lim G'` with direct system `f` and `f'` respectively, any
family of equivalences `eᵢ : Gᵢ ≅ G'ᵢ` such that `e ∘ f = f' ∘ e` induces an equivalence
`lim G ⟶ lim G'`.
-/
def congr (e : (i : ι) → G i ≃+ G' i)
    (he : ∀ i j h, (e j).toAddMonoidHom.comp (f i j h) = (f' i j h).comp (e i)) :
    DirectLimit G f ≃+ DirectLimit G' f' :=
  AddMonoidHom.toAddEquiv (map (e ·) he)
    (map (fun i ↦ (e i).symm) fun i j h ↦ DFunLike.ext _ _ fun x ↦ by
      /-
        R : Type u_1
        inst✝⁶ : Semiring R
        ι : Type u_2
        inst✝⁵ : Preorder ι
        G : ι → Type u_3
        inst✝⁴ : (i : ι) → AddCommMonoid (G i)
        f : (i j : ι) → LE.le i j → AddMonoidHom (G i) (G j)
        inst✝³ : DecidableEq ι
        P : Type u_4
        inst✝² : AddCommMonoid P
        g : (i : ι) → AddMonoidHom (G i) P
        Hg : ∀ (i j : ι) (hij : LE.le i j) (x : G i), Eq ((g j) ((f i j hij) x)) ((g i …
        G' : ι → Type u_5
        inst✝¹ : (i : ι) → AddCommMonoid (G' i)
        f' : (i j : ι) → LE.le i j → AddMonoidHom (G' i) (G' j)
        G'' : ι → Type u_6
        inst✝ : (i : ι) → AddCommMonoid (G'' i)
        f'' : (i j : ι) → LE.le i j → AddMonoidHom (G'' i) (G'' j)
        e : (i : ι) → AddEquiv (G i) (G' i)
        he : ∀ (i j : ι) (h : LE.le i j), Eq ((e j).toAddMonoidHom.comp (f i j h)) ((f …
        i j : ι
        h : LE.le i j
        x : G' i
        ⊢ Eq ((((fun i => ↑(e i).symm) j).comp (f' i j h)) x) (((f i j h).comp ((fun i …
      -/
      have eq1 := DFunLike.congr_fun (he i j h) ((e i).symm x)
      simp only [AddMonoidHom.coe_comp, AddEquiv.coe_toAddMonoidHom, Function.comp_apply,
        AddMonoidHom.coe_coe, AddEquiv.apply_symm_apply] at eq1 ⊢
      /-
        R : Type u_1
        inst✝⁶ : Semiring R
        ι : Type u_2
        inst✝⁵ : Preorder ι
        G : ι → Type u_3
        inst✝⁴ : (i : ι) → AddCommMonoid (G i)
        f : (i j : ι) → LE.le i j → AddMonoidHom (G i) (G j)
        inst✝³ : DecidableEq ι
        P : Type u_4
        inst✝² : AddCommMonoid P
        g : (i : ι) → AddMonoidHom (G i) P
        Hg : ∀ (i j : ι) (hij : LE.le i j) (x : G i), Eq ((g j) ((f i j hij) x)) ((g i …
        G' : ι → Type u_5
        inst✝¹ : (i : ι) → AddCommMonoid (G' i)
        f' : (i j : ι) → LE.le i j → AddMonoidHom (G' i) (G' j)
        G'' : ι → Type u_6
        inst✝ : (i : ι) → AddCommMonoid (G'' i)
        f'' : (i j : ι) → LE.le i j → AddMonoidHom (G'' i) (G'' j)
        e : (i : ι) → AddEquiv (G i) (G' i)
        he : ∀ (i j : ι) (h : LE.le i j), Eq ((e j).toAddMonoidHom.comp (f i j h)) ((f …
        i j : ι
        h : LE.le i j
        x : G' i
        eq1 : Eq ((e j) ((f i j h) ((e i).symm x))) ((f' i j h) x)
        ⊢ Eq ((e j).symm ((f' i j h) x)) ((f i j h) ((e i).symm x))
      -/
      simp [← eq1, of_f])
      /-
        🎉 no goals
      -/
        /-
          R : Type u_1
          inst✝⁶ : Semiring R
          ι : Type u_2
          inst✝⁵ : Preorder ι
          G : ι → Type u_3
          inst✝⁴ : (i : ι) → AddCommMonoid (G i)
          f : (i j : ι) → LE.le i j → AddMonoidHom (G i) (G j)
          inst✝³ : DecidableEq ι
          P : Type u_4
          inst✝² : AddCommMonoid P
          g : (i : ι) → AddMonoidHom (G i) P
          Hg : ∀ (i j : ι) (hij : LE.le i j) (x : G i), Eq ((g j) ((f i j hij) x)) ((g i …
          G' : ι → Type u_5
          inst✝¹ : (i : ι) → AddCommMonoid (G' i)
          f' : (i j : ι) → LE.le i j → AddMonoidHom (G' i) (G' j)
          G'' : ι → Type u_6
          inst✝ : (i : ι) → AddCommMonoid (G'' i)
          f'' : (i j : ι) → LE.le i j → AddMonoidHom (G'' i) (G'' j)
          e : (i : ι) → AddEquiv (G i) (G' i)
          he : ∀ (i j : ι) (h : LE.le i j), Eq ((e j).toAddMonoidHom.comp (f i j h)) ((f …
          ⊢ Eq ((AddCommGroup.DirectLimit.map (fun i => ↑(e i).symm) ⋯).comp (AddCommGro …
        -/
        /-
          🎉 no goals
        -/
    (by simp [map_comp]) (by simp [map_comp])
                             /-
                               🎉 no goals
                             -/


lemma congr_apply_of (e : (i : ι) → G i ≃+ G' i)
    (he : ∀ i j h, (e j).toAddMonoidHom.comp (f i j h) = (f' i j h).comp (e i))
    {i : ι} (g : G i) :
    congr e he (of G f i g) = of G' f' i (e i g) :=
  map_apply_of _ he _


lemma congr_symm_apply_of (e : (i : ι) → G i ≃+ G' i)
    (he : ∀ i j h, (e j).toAddMonoidHom.comp (f i j h) = (f' i j h).comp (e i))
    {i : ι} (g : G' i) :
    (congr e he).symm (of G' f' i g) = of G f i ((e i).symm g) := by
  /-
    ι : Type u_2
    inst✝³ : Preorder ι
    G : ι → Type u_3
    inst✝² : (i : ι) → AddCommMonoid (G i)
    f : (i j : ι) → LE.le i j → AddMonoidHom (G i) (G j)
    inst✝¹ : DecidableEq ι
    G' : ι → Type u_5
    inst✝ : (i : ι) → AddCommMonoid (G' i)
    f' : (i j : ι) → LE.le i j → AddMonoidHom (G' i) (G' j)
    e : (i : ι) → AddEquiv (G i) (G' i)
    he : ∀ (i j : ι) (h : LE.le i j), Eq ((e j).toAddMonoidHom.comp (f i j h)) ((f …
    i : ι
    g : G' i
    ⊢ Eq ((AddCommGroup.DirectLimit.congr e he).symm ((AddCommGroup.DirectLimit.of …
  -/
  simp only [congr, AddMonoidHom.toAddEquiv_symm_apply, map_apply_of, AddMonoidHom.coe_coe]
  /-
    🎉 no goals
  -/


