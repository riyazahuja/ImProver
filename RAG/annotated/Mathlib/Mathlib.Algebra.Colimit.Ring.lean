/-- The direct limit of a directed system is the rings glued together along the maps. -/
def DirectLimit : Type _ :=
  FreeCommRing (Σ i, G i) ⧸
    Ideal.span
      { a |
        (∃ i j H x, of (⟨j, f i j H x⟩ : Σ i, G i) - of ⟨i, x⟩ = a) ∨
          (∃ i, of (⟨i, 1⟩ : Σ i, G i) - 1 = a) ∨
            (∃ i x y, of (⟨i, x + y⟩ : Σ i, G i) - (of ⟨i, x⟩ + of ⟨i, y⟩) = a) ∨
              ∃ i x y, of (⟨i, x * y⟩ : Σ i, G i) - of ⟨i, x⟩ * of ⟨i, y⟩ = a }


instance commRing : CommRing (DirectLimit G f) :=
  Ideal.Quotient.commRing _


instance ring : Ring (DirectLimit G f) :=
  CommRing.toRing

-- Porting note: Added a `Zero` instance to get rid of `0` errors.

instance zero : Zero (DirectLimit G f) := by
  /-
    ι : Type u_1
    inst✝¹ : Preorder ι
    G : ι → Type u_2
    inst✝ : (i : ι) → CommRing (G i)
    f : (i j : ι) → LE.le i j → G i → G j
    ⊢ Zero (Ring.DirectLimit G f)
  -/
  unfold DirectLimit
  /-
    ι : Type u_1
    inst✝¹ : Preorder ι
    G : ι → Type u_2
    inst✝ : (i : ι) → CommRing (G i)
    f : (i j : ι) → LE.le i j → G i → G j
    ⊢ Zero (HasQuotient.Quotient (FreeCommRing (Sigma fun i => G i)) (Ideal.span ( …
  -/
  exact ⟨0⟩
  /-
    🎉 no goals
  -/


instance : Inhabited (DirectLimit G f) :=
  ⟨0⟩


/-- The canonical map from a component to the direct limit. -/
nonrec def of (i) : G i →+* DirectLimit G f :=
  RingHom.mk'
    { toFun := fun x ↦ Ideal.Quotient.mk _ (of (⟨i, x⟩ : Σ i, G i))
      map_one' := Ideal.Quotient.eq.2 <| subset_span <| Or.inr <| Or.inl ⟨i, rfl⟩
      map_mul' := fun x y ↦
        Ideal.Quotient.eq.2 <| subset_span <| Or.inr <| Or.inr <| Or.inr ⟨i, x, y, rfl⟩ }
    fun x y ↦ Ideal.Quotient.eq.2 <| subset_span <| Or.inr <| Or.inr <| Or.inl ⟨i, x, y, rfl⟩


theorem quotientMk_of (i x) : Ideal.Quotient.mk _ (.of ⟨i, x⟩) = of G f i x :=
  rfl


@[simp] theorem of_f {i j} (hij) (x) : of G f j (f i j hij x) = of G f i x :=
  Ideal.Quotient.eq.2 <| subset_span <| Or.inl ⟨i, j, hij, x, rfl⟩


/-- Every element of the direct limit corresponds to some element in
some component of the directed system. -/
theorem exists_of [Nonempty ι] [IsDirected ι (· ≤ ·)] (z : DirectLimit G f) :
    ∃ i x, of G f i x = z := by
  /-
    ι : Type u_1
    inst✝³ : Preorder ι
    G : ι → Type u_2
    inst✝² : (i : ι) → CommRing (G i)
    f : (i j : ι) → LE.le i j → G i → G j
    inst✝¹ : Nonempty ι
    inst✝ : IsDirected ι fun x1 x2 => LE.le x1 x2
    z : Ring.DirectLimit G f
    ⊢ Exists fun i => Exists fun x => Eq ((Ring.DirectLimit.of G f i) x) z
  -/
  obtain ⟨z, rfl⟩ := Ideal.Quotient.mk_surjective z
  /-
    case intro
    ι : Type u_1
    inst✝³ : Preorder ι
    G : ι → Type u_2
    inst✝² : (i : ι) → CommRing (G i)
    f : (i j : ι) → LE.le i j → G i → G j
    inst✝¹ : Nonempty ι
    inst✝ : IsDirected ι fun x1 x2 => LE.le x1 x2
    z : FreeCommRing (Sigma fun i => G i)
    ⊢ Exists fun i => Exists fun x => Eq ((Ring.DirectLimit.of G f i) x) ((Ideal.Q …
  -/
  refine z.induction_on ⟨Classical.arbitrary ι, -1, by simp⟩ (fun ⟨i, x⟩ ↦ ⟨i, x, rfl⟩) ?_ ?_ <;>
    /-
      case intro.refine_1
      ι : Type u_1
      inst✝³ : Preorder ι
      G : ι → Type u_2
      inst✝² : (i : ι) → CommRing (G i)
      f : (i j : ι) → LE.le i j → G i → G j
      inst✝¹ : Nonempty ι
      inst✝ : IsDirected ι fun x1 x2 => LE.le x1 x2
      z : FreeCommRing (Sigma fun i => G i)
      ⊢ ∀ (x y : FreeCommRing (Sigma fun i => G i)), (Exists fun i => Exists fun x_1 …
    -/
    rintro x' y' ⟨i, x, hx⟩ ⟨j, y, hy⟩ <;> have ⟨k, hik, hjk⟩ := exists_ge_ge i j
    /-
      case intro.refine_1.intro.intro.intro.intro
      ι : Type u_1
      inst✝³ : Preorder ι
      G : ι → Type u_2
      inst✝² : (i : ι) → CommRing (G i)
      f : (i j : ι) → LE.le i j → G i → G j
      inst✝¹ : Nonempty ι
      inst✝ : IsDirected ι fun x1 x2 => LE.le x1 x2
      z : FreeCommRing (Sigma fun i => G i)
      x' y' : FreeCommRing (Sigma fun i => G i)
      i : ι
      x : G i
      hx : Eq ((Ring.DirectLimit.of G f i) x) ((Ideal.Quotient.mk (Ideal.span (setOf …
      j : ι
      y : G j
      hy : Eq ((Ring.DirectLimit.of G f j) y) ((Ideal.Quotient.mk (Ideal.span (setOf …
      k : ι
      hik : LE.le i k
      hjk : LE.le j k
      ⊢ Exists fun i => Exists fun x => Eq ((Ring.DirectLimit.of G f i) x) ((Ideal.Q …
    -/
  · exact ⟨k, f i k hik x + f j k hjk y, by rw [map_add, of_f, of_f, hx, hy]; rfl⟩
    /-
      🎉 no goals
    -/
    /-
      case intro.refine_2.intro.intro.intro.intro
      ι : Type u_1
      inst✝³ : Preorder ι
      G : ι → Type u_2
      inst✝² : (i : ι) → CommRing (G i)
      f : (i j : ι) → LE.le i j → G i → G j
      inst✝¹ : Nonempty ι
      inst✝ : IsDirected ι fun x1 x2 => LE.le x1 x2
      z : FreeCommRing (Sigma fun i => G i)
      x' y' : FreeCommRing (Sigma fun i => G i)
      i : ι
      x : G i
      hx : Eq ((Ring.DirectLimit.of G f i) x) ((Ideal.Quotient.mk (Ideal.span (setOf …
      j : ι
      y : G j
      hy : Eq ((Ring.DirectLimit.of G f j) y) ((Ideal.Quotient.mk (Ideal.span (setOf …
      k : ι
      hik : LE.le i k
      hjk : LE.le j k
      ⊢ Exists fun i => Exists fun x => Eq ((Ring.DirectLimit.of G f i) x) ((Ideal.Q …
    -/
  · exact ⟨k, f i k hik x * f j k hjk y, by rw [map_mul, of_f, of_f, hx, hy]; rfl⟩
    /-
      🎉 no goals
    -/


nonrec theorem Polynomial.exists_of [Nonempty ι] [IsDirected ι (· ≤ ·)]
    (q : Polynomial (DirectLimit G fun i j h ↦ f' i j h)) :
    ∃ i p, Polynomial.map (of G (fun i j h ↦ f' i j h) i) p = q :=
  Polynomial.induction_on q
    (fun z ↦
      let ⟨i, x, h⟩ := exists_of z
                  /-
                    ι : Type u_1
                    inst✝³ : Preorder ι
                    G : ι → Type u_2
                    inst✝² : (i : ι) → CommRing (G i)
                    f' : (i j : ι) → LE.le i j → RingHom (G i) (G j)
                    inst✝¹ : Nonempty ι
                    inst✝ : IsDirected ι fun x1 x2 => LE.le x1 x2
                    q : Polynomial (Ring.DirectLimit G fun i j h => ⇑(f' i j h))
                    z : Ring.DirectLimit G fun i j h => ⇑(f' i j h)
                    i : ι
                    x : G i
                    h : Eq ((Ring.DirectLimit.of G (fun i j h => ⇑(f' i j h)) i) x) z
                    ⊢ Eq (Polynomial.map (Ring.DirectLimit.of G (fun i j h => ⇑(f' i j h)) i) (Pol …
                  -/
      ⟨i, C x, by rw [map_C, h]⟩)
                  /-
                    🎉 no goals
                  -/
    (fun q₁ q₂ ⟨i₁, p₁, ih₁⟩ ⟨i₂, p₂, ih₂⟩ ↦
      let ⟨i, h1, h2⟩ := exists_ge_ge i₁ i₂
      ⟨i, p₁.map (f' i₁ i h1) + p₂.map (f' i₂ i h2), by
        /-
          ι : Type u_1
          inst✝³ : Preorder ι
          G : ι → Type u_2
          inst✝² : (i : ι) → CommRing (G i)
          f' : (i j : ι) → LE.le i j → RingHom (G i) (G j)
          inst✝¹ : Nonempty ι
          inst✝ : IsDirected ι fun x1 x2 => LE.le x1 x2
          q : Polynomial (Ring.DirectLimit G fun i j h => ⇑(f' i j h))
          q₁ q₂ : Polynomial (Ring.DirectLimit G fun i j h => ⇑(f' i j h))
          x✝¹ : Exists fun i => Exists fun p => Eq (Polynomial.map (Ring.DirectLimit.of  …
          x✝ : Exists fun i => Exists fun p => Eq (Polynomial.map (Ring.DirectLimit.of G …
          i₁ : ι
          p₁ : Polynomial (G i₁)
          ih₁ : Eq (Polynomial.map (Ring.DirectLimit.of G (fun i j h => ⇑(f' i j h)) i₁) …
          i₂ : ι
          p₂ : Polynomial (G i₂)
          ih₂ : Eq (Polynomial.map (Ring.DirectLimit.of G (fun i j h => ⇑(f' i j h)) i₂) …
          i : ι
          h1 : LE.le i₁ i
          h2 : LE.le i₂ i
          ⊢ Eq (Polynomial.map (Ring.DirectLimit.of G (fun i j h => ⇑(f' i j h)) i) (HAd …
        -/
        rw [Polynomial.map_add, map_map, map_map, ← ih₁, ← ih₂]
        /-
          ι : Type u_1
          inst✝³ : Preorder ι
          G : ι → Type u_2
          inst✝² : (i : ι) → CommRing (G i)
          f' : (i j : ι) → LE.le i j → RingHom (G i) (G j)
          inst✝¹ : Nonempty ι
          inst✝ : IsDirected ι fun x1 x2 => LE.le x1 x2
          q : Polynomial (Ring.DirectLimit G fun i j h => ⇑(f' i j h))
          q₁ q₂ : Polynomial (Ring.DirectLimit G fun i j h => ⇑(f' i j h))
          x✝¹ : Exists fun i => Exists fun p => Eq (Polynomial.map (Ring.DirectLimit.of  …
          x✝ : Exists fun i => Exists fun p => Eq (Polynomial.map (Ring.DirectLimit.of G …
          i₁ : ι
          p₁ : Polynomial (G i₁)
          ih₁ : Eq (Polynomial.map (Ring.DirectLimit.of G (fun i j h => ⇑(f' i j h)) i₁) …
          i₂ : ι
          p₂ : Polynomial (G i₂)
          ih₂ : Eq (Polynomial.map (Ring.DirectLimit.of G (fun i j h => ⇑(f' i j h)) i₂) …
          i : ι
          h1 : LE.le i₁ i
          h2 : LE.le i₂ i
          ⊢ Eq (HAdd.hAdd (Polynomial.map ((Ring.DirectLimit.of G (fun i j h => ⇑(f' i j …
        -/
                              /-
                                🎉 no goals
                              -/
        congr 2 <;> ext x <;> simp_rw [RingHom.comp_apply, of_f]⟩)
                              /-
                                🎉 no goals
                              -/
    fun n z _ ↦
    let ⟨i, x, h⟩ := exists_of z
                              /-
                                ι : Type u_1
                                inst✝³ : Preorder ι
                                G : ι → Type u_2
                                inst✝² : (i : ι) → CommRing (G i)
                                f' : (i j : ι) → LE.le i j → RingHom (G i) (G j)
                                inst✝¹ : Nonempty ι
                                inst✝ : IsDirected ι fun x1 x2 => LE.le x1 x2
                                q : Polynomial (Ring.DirectLimit G fun i j h => ⇑(f' i j h))
                                n : Nat
                                z : Ring.DirectLimit G fun i j h => ⇑(f' i j h)
                                x✝ : Exists fun i => Exists fun p => Eq (Polynomial.map (Ring.DirectLimit.of G …
                                i : ι
                                x : G i
                                h : Eq ((Ring.DirectLimit.of G (fun i j h => ⇑(f' i j h)) i) x) z
                                ⊢ Eq (Polynomial.map (Ring.DirectLimit.of G (fun i j h => ⇑(f' i j h)) i) (HMu …
                              -/
    ⟨i, C x * X ^ (n + 1), by rw [Polynomial.map_mul, map_C, h, Polynomial.map_pow, map_X]⟩
                              /-
                                🎉 no goals
                              -/


@[elab_as_elim]
theorem induction_on [Nonempty ι] [IsDirected ι (· ≤ ·)] {C : DirectLimit G f → Prop}
    (z : DirectLimit G f) (ih : ∀ i x, C (of G f i x)) : C z :=
  let ⟨i, x, hx⟩ := exists_of z
  hx ▸ ih i x


variable (G f) in
/-- The universal property of the direct limit: maps from the components to another ring
that respect the directed system structure (i.e. make some diagram commute) give rise
to a unique map out of the direct limit.
-/
def lift (g : ∀ i, G i →+* P) (Hg : ∀ i j hij x, g j (f i j hij x) = g i x) :
    DirectLimit G f →+* P :=
  Ideal.Quotient.lift _ (FreeCommRing.lift fun x : Σ i, G i ↦ g x.1 x.2)
    (by
      suffices Ideal.span _ ≤
          Ideal.comap (FreeCommRing.lift fun x : Σ i : ι, G i ↦ g x.fst x.snd) ⊥ by
        intro x hx
        exact (mem_bot P).1 (this hx)
      /-
        ι : Type u_1
        inst✝² : Preorder ι
        G : ι → Type u_2
        inst✝¹ : (i : ι) → CommRing (G i)
        f : (i j : ι) → LE.le i j → G i → G j
        P : Type u_3
        inst✝ : CommRing P
        g : (i : ι) → RingHom (G i) P
        Hg : ∀ (i j : ι) (hij : LE.le i j) (x : G i), Eq ((g j) (f i j hij x)) ((g i) x)
        ⊢ LE.le (Ideal.span (setOf fun a => Or (Exists fun i => Exists fun j => Exists …
      -/
      rw [Ideal.span_le]
      /-
        ι : Type u_1
        inst✝² : Preorder ι
        G : ι → Type u_2
        inst✝¹ : (i : ι) → CommRing (G i)
        f : (i j : ι) → LE.le i j → G i → G j
        P : Type u_3
        inst✝ : CommRing P
        g : (i : ι) → RingHom (G i) P
        Hg : ∀ (i j : ι) (hij : LE.le i j) (x : G i), Eq ((g j) (f i j hij x)) ((g i) x)
        ⊢ HasSubset.Subset (setOf fun a => Or (Exists fun i => Exists fun j => Exists  …
      -/
      intro x hx
      /-
        ι : Type u_1
        inst✝² : Preorder ι
        G : ι → Type u_2
        inst✝¹ : (i : ι) → CommRing (G i)
        f : (i j : ι) → LE.le i j → G i → G j
        P : Type u_3
        inst✝ : CommRing P
        g : (i : ι) → RingHom (G i) P
        Hg : ∀ (i j : ι) (hij : LE.le i j) (x : G i), Eq ((g j) (f i j hij x)) ((g i) x)
        x : FreeCommRing (Sigma fun i => G i)
        hx : Membership.mem (setOf fun a => Or (Exists fun i => Exists fun j => Exists …
        ⊢ Membership.mem (↑(Ideal.comap (FreeCommRing.lift fun x => (g x.fst) x.snd) B …
      -/
      rw [SetLike.mem_coe, Ideal.mem_comap, mem_bot]
      /-
        ι : Type u_1
        inst✝² : Preorder ι
        G : ι → Type u_2
        inst✝¹ : (i : ι) → CommRing (G i)
        f : (i j : ι) → LE.le i j → G i → G j
        P : Type u_3
        inst✝ : CommRing P
        g : (i : ι) → RingHom (G i) P
        Hg : ∀ (i j : ι) (hij : LE.le i j) (x : G i), Eq ((g j) (f i j hij x)) ((g i) x)
        x : FreeCommRing (Sigma fun i => G i)
        hx : Membership.mem (setOf fun a => Or (Exists fun i => Exists fun j => Exists …
        ⊢ Eq ((FreeCommRing.lift fun x => (g x.fst) x.snd) x) 0
      -/
      rcases hx with (⟨i, j, hij, x, rfl⟩ | ⟨i, rfl⟩ | ⟨i, x, y, rfl⟩ | ⟨i, x, y, rfl⟩) <;>
        simp only [RingHom.map_sub, lift_of, Hg, RingHom.map_one, RingHom.map_add, RingHom.map_mul,
          (g i).map_one, (g i).map_add, (g i).map_mul, sub_self])


@[simp] theorem lift_of (i x) : lift G f P g Hg (of G f i x) = g i x :=
  FreeCommRing.lift_of _ _


theorem lift_unique (F : DirectLimit G f →+* P) (x) :
                                                                      /-
                                                                        ι : Type u_1
                                                                        inst✝² : Preorder ι
                                                                        G : ι → Type u_2
                                                                        inst✝¹ : (i : ι) → CommRing (G i)
                                                                        f : (i j : ι) → LE.le i j → G i → G j
                                                                        P : Type u_3
                                                                        inst✝ : CommRing P
                                                                        g : (i : ι) → RingHom (G i) P
                                                                        Hg : ∀ (i j : ι) (hij : LE.le i j) (x : G i), Eq ((g j) (f i j hij x)) ((g i) x)
                                                                        F : RingHom (Ring.DirectLimit G f) P
                                                                        x✝ : Ring.DirectLimit G f
                                                                        i j : ι
                                                                        hij : LE.le i j
                                                                        x : G i
                                                                        ⊢ Eq (((fun i => F.comp (Ring.DirectLimit.of G f i)) j) (f i j hij x)) (((fun  …
                                                                      -/
    F x = lift G f P (fun i ↦ F.comp <| of G f i) (fun i j hij x ↦ by simp) x := by
                                                                      /-
                                                                        🎉 no goals
                                                                      -/
  /-
    ι : Type u_1
    inst✝² : Preorder ι
    G : ι → Type u_2
    inst✝¹ : (i : ι) → CommRing (G i)
    f : (i j : ι) → LE.le i j → G i → G j
    P : Type u_3
    inst✝ : CommRing P
    F : RingHom (Ring.DirectLimit G f) P
    x : Ring.DirectLimit G f
    ⊢ Eq (F x) ((Ring.DirectLimit.lift G f P (fun i => F.comp (Ring.DirectLimit.of …
  -/
  obtain ⟨x, rfl⟩ := Ideal.Quotient.mk_surjective x
  exact x.induction_on (by simp) (fun _ ↦ .symm <| lift_of ..)
    (by simp+contextual) (by simp+contextual)


lemma lift_injective [Nonempty ι] [IsDirected ι (· ≤ ·)]
    (injective : ∀ i, Function.Injective <| g i) :
    Function.Injective (lift G f P g Hg) := by
  /-
    ι : Type u_1
    inst✝⁴ : Preorder ι
    G : ι → Type u_2
    inst✝³ : (i : ι) → CommRing (G i)
    f : (i j : ι) → LE.le i j → G i → G j
    P : Type u_3
    inst✝² : CommRing P
    g : (i : ι) → RingHom (G i) P
    Hg : ∀ (i j : ι) (hij : LE.le i j) (x : G i), Eq ((g j) (f i j hij x)) ((g i) x)
    inst✝¹ : Nonempty ι
    inst✝ : IsDirected ι fun x1 x2 => LE.le x1 x2
    injective : ∀ (i : ι), Function.Injective ⇑(g i)
    ⊢ Function.Injective ⇑(Ring.DirectLimit.lift G f P g Hg)
  -/
  simp_rw [injective_iff_map_eq_zero] at injective ⊢
  /-
    ι : Type u_1
    inst✝⁴ : Preorder ι
    G : ι → Type u_2
    inst✝³ : (i : ι) → CommRing (G i)
    f : (i j : ι) → LE.le i j → G i → G j
    P : Type u_3
    inst✝² : CommRing P
    g : (i : ι) → RingHom (G i) P
    Hg : ∀ (i j : ι) (hij : LE.le i j) (x : G i), Eq ((g j) (f i j hij x)) ((g i) x)
    inst✝¹ : Nonempty ι
    inst✝ : IsDirected ι fun x1 x2 => LE.le x1 x2
    injective : ∀ (i : ι) (a : G i), Eq ((g i) a) 0 → Eq a 0
    ⊢ ∀ (a : Ring.DirectLimit G f), Eq ((Ring.DirectLimit.lift G f P g Hg) a) 0 →  …
  -/
  intros z hz
  induction z using DirectLimit.induction_on with
  | ih _ g => rw [lift_of] at hz; rw [injective _ g hz, _root_.map_zero]


open _root_.DirectLimit in
/-- The direct limit constructed as a quotient of the free commutative ring is isomorphic to
the direct limit constructed as a quotient of the disjoint union. -/
def ringEquiv [Nonempty ι] : DirectLimit G (f' · · ·) ≃+* _root_.DirectLimit G f' :=
  .ofRingHom (lift _ _ _ (Ring.of _ _) fun _ _ _ _ ↦ .symm <| eq_of_le ..)
    (Ring.lift _ _ _ (of _ _) fun _ _ _ _ ↦ of_f ..)
        /-
          ι : Type u_1
          inst✝⁵ : Preorder ι
          G : ι → Type u_2
          inst✝⁴ : (i : ι) → CommRing (G i)
          f : (i j : ι) → LE.le i j → G i → G j
          P : Type u_3
          inst✝³ : CommRing P
          g : (i : ι) → RingHom (G i) P
          Hg : ∀ (i j : ι) (hij : LE.le i j) (x : G i), Eq ((g j) (f i j hij x)) ((g i) x)
          f' : (i j : ι) → LE.le i j → RingHom (G i) (G j)
          inst✝² : DirectedSystem G fun i j h => ⇑(f' i j h)
          inst✝¹ : IsDirected ι fun x1 x2 => LE.le x1 x2
          inst✝ : Nonempty ι
          ⊢ Eq ((Ring.DirectLimit.lift G (fun x1 x2 x3 => ⇑(f' x1 x2 x3)) (_root_.Direct …
        -/
    (by ext ⟨_⟩; rw [← Quotient.mk]; simp [Ring.lift, _root_.DirectLimit.lift_def]; rfl)
                                                                                    /-
                                                                                      🎉 no goals
                                                                                    -/
        /-
          ι : Type u_1
          inst✝⁵ : Preorder ι
          G : ι → Type u_2
          inst✝⁴ : (i : ι) → CommRing (G i)
          f : (i j : ι) → LE.le i j → G i → G j
          P : Type u_3
          inst✝³ : CommRing P
          g : (i : ι) → RingHom (G i) P
          Hg : ∀ (i j : ι) (hij : LE.le i j) (x : G i), Eq ((g j) (f i j hij x)) ((g i) x)
          f' : (i j : ι) → LE.le i j → RingHom (G i) (G j)
          inst✝² : DirectedSystem G fun i j h => ⇑(f' i j h)
          inst✝¹ : IsDirected ι fun x1 x2 => LE.le x1 x2
          inst✝ : Nonempty ι
          ⊢ Eq ((DirectLimit.Ring.lift G f' (Ring.DirectLimit G fun x1 x2 x3 => ⇑(f' x1  …
        -/
    (by ext x; exact x.induction_on fun i x ↦ by simp)
               /-
                 🎉 no goals
               -/


theorem ringEquiv_of [Nonempty ι] {i g} : ringEquiv G f' (of _ _ i g) = ⟦⟨i, g⟩⟧ := by
  /-
    ι : Type u_1
    inst✝⁴ : Preorder ι
    G : ι → Type u_2
    inst✝³ : (i : ι) → CommRing (G i)
    f' : (i j : ι) → LE.le i j → RingHom (G i) (G j)
    inst✝² : DirectedSystem G fun i j h => ⇑(f' i j h)
    inst✝¹ : IsDirected ι fun x1 x2 => LE.le x1 x2
    inst✝ : Nonempty ι
    i : ι
    g : G i
    ⊢ Eq ((Ring.DirectLimit.ringEquiv G f') ((Ring.DirectLimit.of G (fun x1 x2 x3  …
  -/
  simp [ringEquiv]; rfl
                    /-
                      🎉 no goals
                    -/


theorem ringEquiv_symm_mk [Nonempty ι] {g} : (ringEquiv G f').symm ⟦g⟧ = of _ _ g.1 g.2 := rfl


/-- A component that corresponds to zero in the direct limit is already zero in some
bigger module in the directed system. -/
theorem of.zero_exact {i x} (hix : of G (f' · · ·) i x = 0) :
    ∃ (j : _) (hij : i ≤ j), f' i j hij x = 0 := by
  /-
    ι : Type u_1
    inst✝³ : Preorder ι
    G : ι → Type u_2
    inst✝² : (i : ι) → CommRing (G i)
    f' : (i j : ι) → LE.le i j → RingHom (G i) (G j)
    inst✝¹ : DirectedSystem G fun i j h => ⇑(f' i j h)
    inst✝ : IsDirected ι fun x1 x2 => LE.le x1 x2
    i : ι
    x : G i
    hix : Eq ((Ring.DirectLimit.of G (fun x1 x2 x3 => ⇑(f' x1 x2 x3)) i) x) 0
    ⊢ Exists fun j => Exists fun hij => Eq ((f' i j hij) x) 0
  -/
  have := Nonempty.intro i
  /-
    ι : Type u_1
    inst✝³ : Preorder ι
    G : ι → Type u_2
    inst✝² : (i : ι) → CommRing (G i)
    f' : (i j : ι) → LE.le i j → RingHom (G i) (G j)
    inst✝¹ : DirectedSystem G fun i j h => ⇑(f' i j h)
    inst✝ : IsDirected ι fun x1 x2 => LE.le x1 x2
    i : ι
    x : G i
    hix : Eq ((Ring.DirectLimit.of G (fun x1 x2 x3 => ⇑(f' x1 x2 x3)) i) x) 0
    this : Nonempty ι
    ⊢ Exists fun j => Exists fun hij => Eq ((f' i j hij) x) 0
  -/
  apply_fun ringEquiv _ _ at hix
  /-
    ι : Type u_1
    inst✝³ : Preorder ι
    G : ι → Type u_2
    inst✝² : (i : ι) → CommRing (G i)
    f' : (i j : ι) → LE.le i j → RingHom (G i) (G j)
    inst✝¹ : DirectedSystem G fun i j h => ⇑(f' i j h)
    inst✝ : IsDirected ι fun x1 x2 => LE.le x1 x2
    i : ι
    x : G i
    this : Nonempty ι
    hix : Eq ((Ring.DirectLimit.ringEquiv G f') ((Ring.DirectLimit.of G (fun x1 x2 …
    ⊢ Exists fun j => Exists fun hij => Eq ((f' i j hij) x) 0
  -/
  rwa [map_zero, ringEquiv_of, DirectLimit.exists_eq_zero] at hix
  /-
    🎉 no goals
  -/


/-- If the maps in the directed system are injective, then the canonical maps
from the components to the direct limits are injective. -/
theorem of_injective [IsDirected ι (· ≤ ·)] [DirectedSystem G fun i j h ↦ f' i j h]
    (hf : ∀ i j hij, Function.Injective (f' i j hij)) (i) :
    Function.Injective (of G (fun i j h ↦ f' i j h) i) :=
  have := Nonempty.intro i
  ((ringEquiv _ _).comp_injective _).mp
                                                       /-
                                                         ι : Type u_1
                                                         inst✝³ : Preorder ι
                                                         G : ι → Type u_2
                                                         inst✝² : (i : ι) → CommRing (G i)
                                                         f' : (i j : ι) → LE.le i j → RingHom (G i) (G j)
                                                         inst✝¹ : IsDirected ι fun x1 x2 => LE.le x1 x2
                                                         inst✝ : DirectedSystem G fun i j h => ⇑(f' i j h)
                                                         hf : ∀ (i j : ι) (hij : LE.le i j), Function.Injective ⇑(f' i j hij)
                                                         i : ι
                                                         this : Nonempty ι
                                                         x✝¹ x✝ : G i
                                                         eq : Eq (Function.comp (⇑(Ring.DirectLimit.ringEquiv G f').toEquiv) (⇑(Ring.Di …
                                                         ⊢ Eq ((fun x => Quotient.mk (DirectLimit.setoid f') ⟨i, x⟩) x✝¹) ((fun x => Qu …
                                                       -/
    fun _ _ eq ↦  DirectLimit.mk_injective f' hf _ (by simpa only [← ringEquiv_of])
                                                       /-
                                                         🎉 no goals
                                                       -/


/--
Consider direct limits `lim G` and `lim G'` with direct system `f` and `f'` respectively, any
family of ring homomorphisms `gᵢ : Gᵢ ⟶ G'ᵢ` such that `g ∘ f = f' ∘ g` induces a ring
homomorphism `lim G ⟶ lim G'`.
-/
def map (g : (i : ι) → G i →+* G' i)
    (hg : ∀ i j h, (g j).comp (f i j h) = (f' i j h).comp (g i)) :
    DirectLimit G (fun _ _ h ↦ f _ _ h) →+* DirectLimit G' fun _ _ h ↦ f' _ _ h :=
  lift _ _ _ (fun i ↦ (of _ _ _).comp (g i)) fun i j h g ↦ by
      /-
        ι : Type u_1
        inst✝⁴ : Preorder ι
        G : ι → Type u_2
        inst✝³ : (i : ι) → CommRing (G i)
        f✝ : (i j : ι) → LE.le i j → G i → G j
        P : Type u_3
        inst✝² : CommRing P
        g✝¹ : (i : ι) → RingHom (G i) P
        Hg : ∀ (i j : ι) (hij : LE.le i j) (x : G i), Eq ((g✝¹ j) (f✝ i j hij x)) ((g✝ …
        f'✝ f : (i j : ι) → LE.le i j → RingHom (G i) (G j)
        G' : ι → Type u_4
        inst✝¹ : (i : ι) → CommRing (G' i)
        f' : (i j : ι) → LE.le i j → RingHom (G' i) (G' j)
        G'' : ι → Type u_5
        inst✝ : (i : ι) → CommRing (G'' i)
        f'' : (i j : ι) → LE.le i j → RingHom (G'' i) (G'' j)
        g✝ : (i : ι) → RingHom (G i) (G' i)
        hg : ∀ (i j : ι) (h : LE.le i j), Eq ((g✝ j).comp (f i j h)) ((f' i j h).comp  …
        i j : ι
        h : LE.le i j
        g : G i
        ⊢ Eq (((fun i => (Ring.DirectLimit.of G' (fun x x_1 h => ⇑(f' x x_1 h)) i).com …
      -/
      have eq1 := DFunLike.congr_fun (hg i j h) g
      /-
        ι : Type u_1
        inst✝⁴ : Preorder ι
        G : ι → Type u_2
        inst✝³ : (i : ι) → CommRing (G i)
        f✝ : (i j : ι) → LE.le i j → G i → G j
        P : Type u_3
        inst✝² : CommRing P
        g✝¹ : (i : ι) → RingHom (G i) P
        Hg : ∀ (i j : ι) (hij : LE.le i j) (x : G i), Eq ((g✝¹ j) (f✝ i j hij x)) ((g✝ …
        f'✝ f : (i j : ι) → LE.le i j → RingHom (G i) (G j)
        G' : ι → Type u_4
        inst✝¹ : (i : ι) → CommRing (G' i)
        f' : (i j : ι) → LE.le i j → RingHom (G' i) (G' j)
        G'' : ι → Type u_5
        inst✝ : (i : ι) → CommRing (G'' i)
        f'' : (i j : ι) → LE.le i j → RingHom (G'' i) (G'' j)
        g✝ : (i : ι) → RingHom (G i) (G' i)
        hg : ∀ (i j : ι) (h : LE.le i j), Eq ((g✝ j).comp (f i j h)) ((f' i j h).comp  …
        i j : ι
        h : LE.le i j
        g : G i
        eq1 : Eq (((g✝ j).comp (f i j h)) g) (((f' i j h).comp (g✝ i)) g)
        ⊢ Eq (((fun i => (Ring.DirectLimit.of G' (fun x x_1 h => ⇑(f' x x_1 h)) i).com …
      -/
      simp only [RingHom.coe_comp, Function.comp_apply] at eq1 ⊢
      /-
        ι : Type u_1
        inst✝⁴ : Preorder ι
        G : ι → Type u_2
        inst✝³ : (i : ι) → CommRing (G i)
        f✝ : (i j : ι) → LE.le i j → G i → G j
        P : Type u_3
        inst✝² : CommRing P
        g✝¹ : (i : ι) → RingHom (G i) P
        Hg : ∀ (i j : ι) (hij : LE.le i j) (x : G i), Eq ((g✝¹ j) (f✝ i j hij x)) ((g✝ …
        f'✝ f : (i j : ι) → LE.le i j → RingHom (G i) (G j)
        G' : ι → Type u_4
        inst✝¹ : (i : ι) → CommRing (G' i)
        f' : (i j : ι) → LE.le i j → RingHom (G' i) (G' j)
        G'' : ι → Type u_5
        inst✝ : (i : ι) → CommRing (G'' i)
        f'' : (i j : ι) → LE.le i j → RingHom (G'' i) (G'' j)
        g✝ : (i : ι) → RingHom (G i) (G' i)
        hg : ∀ (i j : ι) (h : LE.le i j), Eq ((g✝ j).comp (f i j h)) ((f' i j h).comp  …
        i j : ι
        h : LE.le i j
        g : G i
        eq1 : Eq ((g✝ j) ((f i j h) g)) ((f' i j h) ((g✝ i) g))
        ⊢ Eq ((Ring.DirectLimit.of G' (fun x x_1 h => ⇑(f' x x_1 h)) j) ((g✝ j) ((f i  …
      -/
      rw [eq1, of_f]
      /-
        🎉 no goals
      -/


@[simp] lemma map_apply_of (g : (i : ι) → G i →+* G' i)
    (hg : ∀ i j h, (g j).comp (f i j h) = (f' i j h).comp (g i))
    {i : ι} (x : G i) :
    map g hg (of G _ _ x) = of G' (fun _ _ h ↦ f' _ _ h) i (g i x) :=
  lift_of _ _ _ _ _


@[simp] lemma map_id :
    map (fun _ ↦ RingHom.id _) (fun _ _ _ ↦ rfl) = RingHom.id (DirectLimit G fun _ _ h ↦ f _ _ h) :=
  DFunLike.ext _ _ fun x ↦ by
    /-
      ι : Type u_1
      inst✝¹ : Preorder ι
      G : ι → Type u_2
      inst✝ : (i : ι) → CommRing (G i)
      f : (i j : ι) → LE.le i j → RingHom (G i) (G j)
      x : Ring.DirectLimit G fun x x_1 h => ⇑(f x x_1 h)
      ⊢ Eq ((Ring.DirectLimit.map (fun x => RingHom.id (G x)) ⋯) x) ((RingHom.id (Ri …
    -/
    obtain ⟨x, rfl⟩ := Ideal.Quotient.mk_surjective x
    /-
      case intro
      ι : Type u_1
      inst✝¹ : Preorder ι
      G : ι → Type u_2
      inst✝ : (i : ι) → CommRing (G i)
      f : (i j : ι) → LE.le i j → RingHom (G i) (G j)
      x : FreeCommRing (Sigma fun i => G i)
      ⊢ Eq ((Ring.DirectLimit.map (fun x => RingHom.id (G x)) ⋯) ((Ideal.Quotient.mk …
    -/
    refine x.induction_on (by simp) (fun _ ↦ ?_) (by simp+contextual) (by simp+contextual)
    /-
      case intro
      ι : Type u_1
      inst✝¹ : Preorder ι
      G : ι → Type u_2
      inst✝ : (i : ι) → CommRing (G i)
      f : (i j : ι) → LE.le i j → RingHom (G i) (G j)
      x : FreeCommRing (Sigma fun i => G i)
      x✝ : Sigma fun i => G i
      ⊢ Eq ((Ring.DirectLimit.map (fun x => RingHom.id (G x)) ⋯) ((Ideal.Quotient.mk …
    -/
    rw [quotientMk_of, map_apply_of]; rfl
                                      /-
                                        🎉 no goals
                                      -/


lemma map_comp (g₁ : (i : ι) → G i →+* G' i) (g₂ : (i : ι) → G' i →+* G'' i)
    (hg₁ : ∀ i j h, (g₁ j).comp (f i j h) = (f' i j h).comp (g₁ i))
    (hg₂ : ∀ i j h, (g₂ j).comp (f' i j h) = (f'' i j h).comp (g₂ i)) :
    ((map g₂ hg₂).comp (map g₁ hg₁) :
      DirectLimit G (fun _ _ h ↦ f _ _ h) →+* DirectLimit G'' fun _ _ h ↦ f'' _ _ h) =
    (map (fun i ↦ (g₂ i).comp (g₁ i)) fun i j h ↦ by
      /-
        ι : Type u_1
        inst✝⁴ : Preorder ι
        G : ι → Type u_2
        inst✝³ : (i : ι) → CommRing (G i)
        f✝ : (i j : ι) → LE.le i j → G i → G j
        P : Type u_3
        inst✝² : CommRing P
        g : (i : ι) → RingHom (G i) P
        Hg : ∀ (i j : ι) (hij : LE.le i j) (x : G i), Eq ((g j) (f✝ i j hij x)) ((g i) …
        f'✝ f : (i j : ι) → LE.le i j → RingHom (G i) (G j)
        G' : ι → Type u_4
        inst✝¹ : (i : ι) → CommRing (G' i)
        f' : (i j : ι) → LE.le i j → RingHom (G' i) (G' j)
        G'' : ι → Type u_5
        inst✝ : (i : ι) → CommRing (G'' i)
        f'' : (i j : ι) → LE.le i j → RingHom (G'' i) (G'' j)
        g₁ : (i : ι) → RingHom (G i) (G' i)
        g₂ : (i : ι) → RingHom (G' i) (G'' i)
        hg₁ : ∀ (i j : ι) (h : LE.le i j), Eq ((g₁ j).comp (f i j h)) ((f' i j h).comp …
        hg₂ : ∀ (i j : ι) (h : LE.le i j), Eq ((g₂ j).comp (f' i j h)) ((f'' i j h).co …
        i j : ι
        h : LE.le i j
        ⊢ Eq (((fun i => (g₂ i).comp (g₁ i)) j).comp (f i j h)) ((f'' i j h).comp ((fu …
      -/
      rw [RingHom.comp_assoc, hg₁ i, ← RingHom.comp_assoc, hg₂ i, RingHom.comp_assoc] :
      /-
        🎉 no goals
      -/
      DirectLimit G (fun _ _ h ↦ f _ _ h) →+* DirectLimit G'' fun _ _ h ↦ f'' _ _ h) :=
  DFunLike.ext _ _ fun x ↦ by
    /-
      ι : Type u_1
      inst✝³ : Preorder ι
      G : ι → Type u_2
      inst✝² : (i : ι) → CommRing (G i)
      f : (i j : ι) → LE.le i j → RingHom (G i) (G j)
      G' : ι → Type u_4
      inst✝¹ : (i : ι) → CommRing (G' i)
      f' : (i j : ι) → LE.le i j → RingHom (G' i) (G' j)
      G'' : ι → Type u_5
      inst✝ : (i : ι) → CommRing (G'' i)
      f'' : (i j : ι) → LE.le i j → RingHom (G'' i) (G'' j)
      g₁ : (i : ι) → RingHom (G i) (G' i)
      g₂ : (i : ι) → RingHom (G' i) (G'' i)
      hg₁ : ∀ (i j : ι) (h : LE.le i j), Eq ((g₁ j).comp (f i j h)) ((f' i j h).comp …
      hg₂ : ∀ (i j : ι) (h : LE.le i j), Eq ((g₂ j).comp (f' i j h)) ((f'' i j h).co …
      x : Ring.DirectLimit G fun x x_1 h => ⇑(f x x_1 h)
      ⊢ Eq (((Ring.DirectLimit.map g₂ hg₂).comp (Ring.DirectLimit.map g₁ hg₁)) x) (( …
    -/
    obtain ⟨x, rfl⟩ := Ideal.Quotient.mk_surjective x
    /-
      case intro
      ι : Type u_1
      inst✝³ : Preorder ι
      G : ι → Type u_2
      inst✝² : (i : ι) → CommRing (G i)
      f : (i j : ι) → LE.le i j → RingHom (G i) (G j)
      G' : ι → Type u_4
      inst✝¹ : (i : ι) → CommRing (G' i)
      f' : (i j : ι) → LE.le i j → RingHom (G' i) (G' j)
      G'' : ι → Type u_5
      inst✝ : (i : ι) → CommRing (G'' i)
      f'' : (i j : ι) → LE.le i j → RingHom (G'' i) (G'' j)
      g₁ : (i : ι) → RingHom (G i) (G' i)
      g₂ : (i : ι) → RingHom (G' i) (G'' i)
      hg₁ : ∀ (i j : ι) (h : LE.le i j), Eq ((g₁ j).comp (f i j h)) ((f' i j h).comp …
      hg₂ : ∀ (i j : ι) (h : LE.le i j), Eq ((g₂ j).comp (f' i j h)) ((f'' i j h).co …
      x : FreeCommRing (Sigma fun i => G i)
      ⊢ Eq (((Ring.DirectLimit.map g₂ hg₂).comp (Ring.DirectLimit.map g₁ hg₁)) ((Ide …
    -/
    refine x.induction_on (by simp) (fun _ ↦ ?_) (by simp+contextual) (by simp+contextual)
    /-
      case intro
      ι : Type u_1
      inst✝³ : Preorder ι
      G : ι → Type u_2
      inst✝² : (i : ι) → CommRing (G i)
      f : (i j : ι) → LE.le i j → RingHom (G i) (G j)
      G' : ι → Type u_4
      inst✝¹ : (i : ι) → CommRing (G' i)
      f' : (i j : ι) → LE.le i j → RingHom (G' i) (G' j)
      G'' : ι → Type u_5
      inst✝ : (i : ι) → CommRing (G'' i)
      f'' : (i j : ι) → LE.le i j → RingHom (G'' i) (G'' j)
      g₁ : (i : ι) → RingHom (G i) (G' i)
      g₂ : (i : ι) → RingHom (G' i) (G'' i)
      hg₁ : ∀ (i j : ι) (h : LE.le i j), Eq ((g₁ j).comp (f i j h)) ((f' i j h).comp …
      hg₂ : ∀ (i j : ι) (h : LE.le i j), Eq ((g₂ j).comp (f' i j h)) ((f'' i j h).co …
      x : FreeCommRing (Sigma fun i => G i)
      x✝ : Sigma fun i => G i
      ⊢ Eq (((Ring.DirectLimit.map g₂ hg₂).comp (Ring.DirectLimit.map g₁ hg₁)) ((Ide …
    -/
    rw [RingHom.comp_apply, quotientMk_of]
    /-
      case intro
      ι : Type u_1
      inst✝³ : Preorder ι
      G : ι → Type u_2
      inst✝² : (i : ι) → CommRing (G i)
      f : (i j : ι) → LE.le i j → RingHom (G i) (G j)
      G' : ι → Type u_4
      inst✝¹ : (i : ι) → CommRing (G' i)
      f' : (i j : ι) → LE.le i j → RingHom (G' i) (G' j)
      G'' : ι → Type u_5
      inst✝ : (i : ι) → CommRing (G'' i)
      f'' : (i j : ι) → LE.le i j → RingHom (G'' i) (G'' j)
      g₁ : (i : ι) → RingHom (G i) (G' i)
      g₂ : (i : ι) → RingHom (G' i) (G'' i)
      hg₁ : ∀ (i j : ι) (h : LE.le i j), Eq ((g₁ j).comp (f i j h)) ((f' i j h).comp …
      hg₂ : ∀ (i j : ι) (h : LE.le i j), Eq ((g₂ j).comp (f' i j h)) ((f'' i j h).co …
      x : FreeCommRing (Sigma fun i => G i)
      x✝ : Sigma fun i => G i
      ⊢ Eq ((Ring.DirectLimit.map g₂ hg₂) ((Ring.DirectLimit.map g₁ hg₁) ((Ring.Dire …
    -/
    simp_rw [map_apply_of]
    /-
      case intro
      ι : Type u_1
      inst✝³ : Preorder ι
      G : ι → Type u_2
      inst✝² : (i : ι) → CommRing (G i)
      f : (i j : ι) → LE.le i j → RingHom (G i) (G j)
      G' : ι → Type u_4
      inst✝¹ : (i : ι) → CommRing (G' i)
      f' : (i j : ι) → LE.le i j → RingHom (G' i) (G' j)
      G'' : ι → Type u_5
      inst✝ : (i : ι) → CommRing (G'' i)
      f'' : (i j : ι) → LE.le i j → RingHom (G'' i) (G'' j)
      g₁ : (i : ι) → RingHom (G i) (G' i)
      g₂ : (i : ι) → RingHom (G' i) (G'' i)
      hg₁ : ∀ (i j : ι) (h : LE.le i j), Eq ((g₁ j).comp (f i j h)) ((f' i j h).comp …
      hg₂ : ∀ (i j : ι) (h : LE.le i j), Eq ((g₂ j).comp (f' i j h)) ((f'' i j h).co …
      x : FreeCommRing (Sigma fun i => G i)
      x✝ : Sigma fun i => G i
      ⊢ Eq ((Ring.DirectLimit.of G'' (fun x x_1 h => ⇑(f'' x x_1 h)) x✝.fst) ((g₂ x✝ …
    -/
    rfl
    /-
      🎉 no goals
    -/


/--
Consider direct limits `lim G` and `lim G'` with direct system `f` and `f'` respectively, any
family of equivalences `eᵢ : Gᵢ ≅ G'ᵢ` such that `e ∘ f = f' ∘ e` induces an equivalence
`lim G ⟶ lim G'`.
-/
def congr (e : (i : ι) → G i ≃+* G' i)
    (he : ∀ i j h, (e j).toRingHom.comp (f i j h) = (f' i j h).comp (e i)) :
    DirectLimit G (fun _ _ h ↦ f _ _ h) ≃+* DirectLimit G' fun _ _ h ↦ f' _ _ h :=
  RingEquiv.ofRingHom
    (map (e ·) he)
    (map (fun i ↦ (e i).symm) fun i j h ↦ DFunLike.ext _ _ fun x ↦ by
      /-
        ι : Type u_1
        inst✝⁴ : Preorder ι
        G : ι → Type u_2
        inst✝³ : (i : ι) → CommRing (G i)
        f✝ : (i j : ι) → LE.le i j → G i → G j
        P : Type u_3
        inst✝² : CommRing P
        g : (i : ι) → RingHom (G i) P
        Hg : ∀ (i j : ι) (hij : LE.le i j) (x : G i), Eq ((g j) (f✝ i j hij x)) ((g i) …
        f'✝ f : (i j : ι) → LE.le i j → RingHom (G i) (G j)
        G' : ι → Type u_4
        inst✝¹ : (i : ι) → CommRing (G' i)
        f' : (i j : ι) → LE.le i j → RingHom (G' i) (G' j)
        G'' : ι → Type u_5
        inst✝ : (i : ι) → CommRing (G'' i)
        f'' : (i j : ι) → LE.le i j → RingHom (G'' i) (G'' j)
        e : (i : ι) → RingEquiv (G i) (G' i)
        he : ∀ (i j : ι) (h : LE.le i j), Eq ((e j).toRingHom.comp (f i j h)) ((f' i j …
        i j : ι
        h : LE.le i j
        x : G' i
        ⊢ Eq ((((fun i => ↑(e i).symm) j).comp (f' i j h)) x) (((f i j h).comp ((fun i …
      -/
      have eq1 := DFunLike.congr_fun (he i j h) ((e i).symm x)
      simp only [RingEquiv.toRingHom_eq_coe, RingHom.coe_comp, RingHom.coe_coe, Function.comp_apply,
        RingEquiv.apply_symm_apply] at eq1 ⊢
      /-
        ι : Type u_1
        inst✝⁴ : Preorder ι
        G : ι → Type u_2
        inst✝³ : (i : ι) → CommRing (G i)
        f✝ : (i j : ι) → LE.le i j → G i → G j
        P : Type u_3
        inst✝² : CommRing P
        g : (i : ι) → RingHom (G i) P
        Hg : ∀ (i j : ι) (hij : LE.le i j) (x : G i), Eq ((g j) (f✝ i j hij x)) ((g i) …
        f'✝ f : (i j : ι) → LE.le i j → RingHom (G i) (G j)
        G' : ι → Type u_4
        inst✝¹ : (i : ι) → CommRing (G' i)
        f' : (i j : ι) → LE.le i j → RingHom (G' i) (G' j)
        G'' : ι → Type u_5
        inst✝ : (i : ι) → CommRing (G'' i)
        f'' : (i j : ι) → LE.le i j → RingHom (G'' i) (G'' j)
        e : (i : ι) → RingEquiv (G i) (G' i)
        he : ∀ (i j : ι) (h : LE.le i j), Eq ((e j).toRingHom.comp (f i j h)) ((f' i j …
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
          ι : Type u_1
          inst✝⁴ : Preorder ι
          G : ι → Type u_2
          inst✝³ : (i : ι) → CommRing (G i)
          f✝ : (i j : ι) → LE.le i j → G i → G j
          P : Type u_3
          inst✝² : CommRing P
          g : (i : ι) → RingHom (G i) P
          Hg : ∀ (i j : ι) (hij : LE.le i j) (x : G i), Eq ((g j) (f✝ i j hij x)) ((g i) …
          f'✝ f : (i j : ι) → LE.le i j → RingHom (G i) (G j)
          G' : ι → Type u_4
          inst✝¹ : (i : ι) → CommRing (G' i)
          f' : (i j : ι) → LE.le i j → RingHom (G' i) (G' j)
          G'' : ι → Type u_5
          inst✝ : (i : ι) → CommRing (G'' i)
          f'' : (i j : ι) → LE.le i j → RingHom (G'' i) (G'' j)
          e : (i : ι) → RingEquiv (G i) (G' i)
          he : ∀ (i j : ι) (h : LE.le i j), Eq ((e j).toRingHom.comp (f i j h)) ((f' i j …
          ⊢ Eq ((Ring.DirectLimit.map (fun x => ↑(e x)) he).comp (Ring.DirectLimit.map ( …
        -/
        /-
          🎉 no goals
        -/
    (by simp [map_comp]) (by simp [map_comp])
                             /-
                               🎉 no goals
                             -/


lemma congr_apply_of (e : (i : ι) → G i ≃+* G' i)
    (he : ∀ i j h, (e j).toRingHom.comp (f i j h) = (f' i j h).comp (e i))
    {i : ι} (g : G i) :
    congr e he (of G _ i g) = of G' (fun _ _ h ↦ f' _ _ h) i (e i g) :=
  map_apply_of _ he _


lemma congr_symm_apply_of (e : (i : ι) → G i ≃+* G' i)
    (he : ∀ i j h, (e j).toRingHom.comp (f i j h) = (f' i j h).comp (e i))
    {i : ι} (g : G' i) :
    (congr e he).symm (of G' _ i g) = of G (fun _ _ h ↦ f _ _ h) i ((e i).symm g) := by
  /-
    ι : Type u_1
    inst✝² : Preorder ι
    G : ι → Type u_2
    inst✝¹ : (i : ι) → CommRing (G i)
    f : (i j : ι) → LE.le i j → RingHom (G i) (G j)
    G' : ι → Type u_4
    inst✝ : (i : ι) → CommRing (G' i)
    f' : (i j : ι) → LE.le i j → RingHom (G' i) (G' j)
    e : (i : ι) → RingEquiv (G i) (G' i)
    he : ∀ (i j : ι) (h : LE.le i j), Eq ((e j).toRingHom.comp (f i j h)) ((f' i j …
    i : ι
    g : G' i
    ⊢ Eq ((Ring.DirectLimit.congr e he).symm ((Ring.DirectLimit.of G' (fun x x_1 h …
  -/
  simp only [congr, RingEquiv.ofRingHom_symm_apply, map_apply_of, RingHom.coe_coe]
  /-
    🎉 no goals
  -/


instance nontrivial [DirectedSystem G (f' · · ·)] :
    Nontrivial (Ring.DirectLimit G (f' · · ·)) :=
  ⟨⟨0, 1,
                        /-
                          ι : Type u_1
                          inst✝⁴ : Preorder ι
                          G : ι → Type u_2
                          inst✝³ : Nonempty ι
                          inst✝² : IsDirected ι fun x1 x2 => LE.le x1 x2
                          inst✝¹ : (i : ι) → Field (G i)
                          f : (i j : ι) → LE.le i j → G i → G j
                          f' : (i j : ι) → LE.le i j → RingHom (G i) (G j)
                          inst✝ : DirectedSystem G fun x1 x2 x3 => ⇑(f' x1 x2 x3)
                          ⊢ Nonempty ι
                        -/
      Nonempty.elim (by infer_instance) fun i : ι ↦ by
                        /-
                          🎉 no goals
                        -/
        /-
          ι : Type u_1
          inst✝⁴ : Preorder ι
          G : ι → Type u_2
          inst✝³ : Nonempty ι
          inst✝² : IsDirected ι fun x1 x2 => LE.le x1 x2
          inst✝¹ : (i : ι) → Field (G i)
          f : (i j : ι) → LE.le i j → G i → G j
          f' : (i j : ι) → LE.le i j → RingHom (G i) (G j)
          inst✝ : DirectedSystem G fun x1 x2 x3 => ⇑(f' x1 x2 x3)
          i : ι
          ⊢ Ne 0 1
        -/
        change (0 : Ring.DirectLimit G (f' · · ·)) ≠ 1
        /-
          ι : Type u_1
          inst✝⁴ : Preorder ι
          G : ι → Type u_2
          inst✝³ : Nonempty ι
          inst✝² : IsDirected ι fun x1 x2 => LE.le x1 x2
          inst✝¹ : (i : ι) → Field (G i)
          f : (i j : ι) → LE.le i j → G i → G j
          f' : (i j : ι) → LE.le i j → RingHom (G i) (G j)
          inst✝ : DirectedSystem G fun x1 x2 x3 => ⇑(f' x1 x2 x3)
          i : ι
          ⊢ Ne 0 1
        -/
        rw [← (Ring.DirectLimit.of _ _ _).map_one]
          /-
            ι : Type u_1
            inst✝⁴ : Preorder ι
            G : ι → Type u_2
            inst✝³ : Nonempty ι
            inst✝² : IsDirected ι fun x1 x2 => LE.le x1 x2
            inst✝¹ : (i : ι) → Field (G i)
            f : (i j : ι) → LE.le i j → G i → G j
            f' : (i j : ι) → LE.le i j → RingHom (G i) (G j)
            inst✝ : DirectedSystem G fun x1 x2 x3 => ⇑(f' x1 x2 x3)
            i : ι
            ⊢ Ne 0 ((Ring.DirectLimit.of G (fun x1 x2 x3 => ⇑(f' x1 x2 x3)) ?m.328179) 1)
          -/
        · intro H; rcases Ring.DirectLimit.of.zero_exact H.symm with ⟨j, hij, hf⟩
          /-
            case intro.intro
            ι : Type u_1
            inst✝⁴ : Preorder ι
            G : ι → Type u_2
            inst✝³ : Nonempty ι
            inst✝² : IsDirected ι fun x1 x2 => LE.le x1 x2
            inst✝¹ : (i : ι) → Field (G i)
            f : (i j : ι) → LE.le i j → G i → G j
            f' : (i j : ι) → LE.le i j → RingHom (G i) (G j)
            inst✝ : DirectedSystem G fun x1 x2 x3 => ⇑(f' x1 x2 x3)
            i : ι
            H : Eq 0 ((Ring.DirectLimit.of G (fun x1 x2 x3 => ⇑(f' x1 x2 x3)) ?m.328179) 1)
            j : ι
            hij : LE.le ?m.328179 j
            hf : Eq ((f' ?m.328179 j hij) 1) 0
            ⊢ False
          -/
          rw [(f' i j hij).map_one] at hf
          /-
            case intro.intro
            ι : Type u_1
            inst✝⁴ : Preorder ι
            G : ι → Type u_2
            inst✝³ : Nonempty ι
            inst✝² : IsDirected ι fun x1 x2 => LE.le x1 x2
            inst✝¹ : (i : ι) → Field (G i)
            f : (i j : ι) → LE.le i j → G i → G j
            f' : (i j : ι) → LE.le i j → RingHom (G i) (G j)
            inst✝ : DirectedSystem G fun x1 x2 x3 => ⇑(f' x1 x2 x3)
            i : ι
            H : Eq 0 ((Ring.DirectLimit.of G (fun x1 x2 x3 => ⇑(f' x1 x2 x3)) i) 1)
            j : ι
            hij : LE.le i j
            hf : Eq 1 0
            ⊢ False
          -/
          exact one_ne_zero hf⟩⟩
          /-
            🎉 no goals
          -/


theorem exists_inv {p : Ring.DirectLimit G f} : p ≠ 0 → ∃ y, p * y = 1 :=
  Ring.DirectLimit.induction_on p fun i x H ↦
    ⟨Ring.DirectLimit.of G f i x⁻¹, by
      rw [← (Ring.DirectLimit.of _ _ _).map_mul,
        mul_inv_cancel₀ fun h : x = 0 ↦ H <| by rw [h, (Ring.DirectLimit.of _ _ _).map_zero],
        (Ring.DirectLimit.of _ _ _).map_one]⟩


open Classical in
/-- Noncomputable multiplicative inverse in a direct limit of fields. -/
noncomputable def inv (p : Ring.DirectLimit G f) : Ring.DirectLimit G f :=
  if H : p = 0 then 0 else Classical.choose (DirectLimit.exists_inv G f H)


protected theorem mul_inv_cancel {p : Ring.DirectLimit G f} (hp : p ≠ 0) : p * inv G f p = 1 := by
  /-
    ι : Type u_1
    inst✝³ : Preorder ι
    G : ι → Type u_2
    inst✝² : Nonempty ι
    inst✝¹ : IsDirected ι fun x1 x2 => LE.le x1 x2
    inst✝ : (i : ι) → Field (G i)
    f : (i j : ι) → LE.le i j → G i → G j
    p : Ring.DirectLimit G f
    hp : Ne p 0
    ⊢ Eq (HMul.hMul p (Field.DirectLimit.inv G f p)) 1
  -/
  rw [inv, dif_neg hp, Classical.choose_spec (DirectLimit.exists_inv G f hp)]
  /-
    🎉 no goals
  -/


protected theorem inv_mul_cancel {p : Ring.DirectLimit G f} (hp : p ≠ 0) : inv G f p * p = 1 := by
  /-
    ι : Type u_1
    inst✝³ : Preorder ι
    G : ι → Type u_2
    inst✝² : Nonempty ι
    inst✝¹ : IsDirected ι fun x1 x2 => LE.le x1 x2
    inst✝ : (i : ι) → Field (G i)
    f : (i j : ι) → LE.le i j → G i → G j
    p : Ring.DirectLimit G f
    hp : Ne p 0
    ⊢ Eq (HMul.hMul (Field.DirectLimit.inv G f p) p) 1
  -/
  rw [_root_.mul_comm, DirectLimit.mul_inv_cancel G f hp]
  /-
    🎉 no goals
  -/


/-- Noncomputable field structure on the direct limit of fields.
See note [reducible non-instances]. -/
protected noncomputable abbrev field [DirectedSystem G (f' · · ·)] :
    Field (Ring.DirectLimit G (f' · · ·)) where
  -- This used to include the parent CommRing and Nontrivial instances,
  -- but leaving them implicit avoids a very expensive (2-3 minutes!) eta expansion.
  inv := inv G (f' · · ·)
  mul_inv_cancel := fun _ ↦ DirectLimit.mul_inv_cancel G (f' · · ·)
  inv_zero := dif_pos rfl
  nnqsmul := _
  nnqsmul_def _ _ := rfl
  qsmul := _
  qsmul_def _ _ := rfl


